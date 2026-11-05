import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace

from pydantic import TypeAdapter

from ..coordination.models import SourceReference
from ..storage.transactions import transaction
from .providers import Availability, Document, SourceResult, SourceStatus, revision


@dataclass(frozen=True)
class SourceChange:
    id: int
    provider: str
    source_id: str
    revision: str
    kind: str
    observed_at: float


class SourceIndex:
    def __init__(
        self,
        db: sqlite3.Connection,
        pet: str,
        changed: Callable[[tuple[SourceChange, ...]], None] | None = None,
    ) -> None:
        self.db = db
        self.pet = pet
        self.changed = changed

    def documents(self, *, history: bool = False) -> tuple[Document, ...]:
        predicate = "" if history else " AND active=1"
        rows = self.db.execute(
            f"SELECT payload,active FROM companion_documents WHERE pet=?{predicate}",
            (self.pet,),
        )
        adapter = TypeAdapter(Document)
        result = []
        for row in rows:
            document = adapter.validate_json(row["payload"])
            if not row["active"]:
                document = replace(
                    document, source=replace(document.source, available=False)
                )
            result.append(document)
        return tuple(result)

    def statuses(self) -> tuple[SourceStatus, ...]:
        rows = self.db.execute(
            "SELECT payload FROM companion_source_status WHERE pet=?", (self.pet,)
        )
        adapter = TypeAdapter(SourceStatus)
        return tuple(adapter.validate_json(row["payload"]) for row in rows)

    def capture(self, result: SourceResult, *, complete: bool | None = None) -> None:
        checkpoint = self.checkpoint()
        provider = result.status.provider
        complete = result.complete if complete is None else complete
        previous_status = next(
            (status for status in self.statuses() if status.provider == provider), None
        )
        if any(document.source.provider != provider for document in result.documents):
            raise ValueError(
                "A source result must retain one provider-qualified identity"
            )
        previous = {
            document.source.id: document
            for document in self.documents()
            if document.source.provider == provider
        }
        adapter = TypeAdapter(Document)
        with transaction(self.db):
            if previous_status is not None and (
                previous_status.availability != result.status.availability
                or previous_status.detail != result.status.detail
            ):
                self._change(
                    SourceReference(
                        f"provider:{provider}",
                        provider,
                        revision(
                            result.status.availability.value + result.status.detail
                        ),
                        result.status.checked_at,
                    ),
                    result.status.availability.value,
                )
            self.db.execute(
                """INSERT INTO companion_source_status(pet,provider,payload)
                VALUES (?,?,?) ON CONFLICT(pet,provider) DO UPDATE SET payload=excluded.payload""",
                (
                    self.pet,
                    provider,
                    TypeAdapter(SourceStatus).dump_json(result.status).decode(),
                ),
            )
            for document in result.documents:
                old = previous.get(document.source.id)
                if old and old.source.revision == document.source.revision:
                    document = replace(
                        document,
                        source=replace(
                            document.source, observed_at=old.source.observed_at
                        ),
                    )
                else:
                    self._change(document.source, "updated" if old else "added")
                self.db.execute(
                    "UPDATE companion_documents SET active=0 WHERE pet=? AND provider=? AND source_id=?",
                    (self.pet, provider, document.source.id),
                )
                self.db.execute(
                    """INSERT INTO companion_documents VALUES (?,?,?,?,?,1,?)
                    ON CONFLICT(pet,provider,source_id,revision) DO UPDATE SET
                    payload=excluded.payload,active=1""",
                    (
                        self.pet,
                        provider,
                        document.source.id,
                        document.source.revision,
                        adapter.dump_json(document).decode(),
                        document.source.observed_at,
                    ),
                )
            if complete and result.status.availability == Availability.AVAILABLE:
                identifiers = {document.source.id for document in result.documents}
                for identifier, old in previous.items():
                    if identifier in identifiers:
                        continue
                    self.db.execute(
                        "UPDATE companion_documents SET active=0 WHERE pet=? AND provider=? AND source_id=?",
                        (self.pet, provider, identifier),
                    )
                    self._change(
                        replace(old.source, observed_at=result.status.checked_at),
                        "removed",
                    )
            if result.status.availability == Availability.AVAILABLE:
                for identifier in result.missing_ids:
                    old = previous.get(identifier)
                    if old is not None:
                        self.db.execute(
                            "UPDATE companion_documents SET active=0 WHERE pet=? AND provider=? AND source_id=?",
                            (self.pet, provider, identifier),
                        )
                        self._change(
                            replace(old.source, observed_at=result.status.checked_at),
                            "removed",
                        )

        self._notify(checkpoint)

    def _notify(self, checkpoint: int) -> None:
        changes = self.changes(checkpoint)
        if changes and self.changed is not None:
            self.changed(changes)

    def _change(self, source: SourceReference, kind: str) -> None:
        self.db.execute(
            """INSERT INTO companion_source_changes
            (pet,provider,source_id,revision,kind,observed_at) VALUES (?,?,?,?,?,?)""",
            (
                self.pet,
                source.provider,
                source.id,
                source.revision,
                kind,
                source.observed_at,
            ),
        )

    def changes(self, after: int = 0) -> tuple[SourceChange, ...]:
        rows = self.db.execute(
            """SELECT id,provider,source_id,revision,kind,observed_at
            FROM companion_source_changes WHERE pet=? AND id>? ORDER BY id""",
            (self.pet, after),
        )
        return tuple(SourceChange(**dict(row)) for row in rows)

    def checkpoint(self) -> int:
        row = self.db.execute(
            "SELECT MAX(id) AS checkpoint FROM companion_source_changes WHERE pet=?",
            (self.pet,),
        ).fetchone()
        return row["checkpoint"] or 0

    def document(self, source_id: str, revision: str | None = None) -> Document | None:
        matches = [
            document
            for document in self.documents(history=revision is not None)
            if document.source.id == source_id
            and (revision is None or document.source.revision == revision)
        ]
        return matches[0] if matches else None

    def current(
        self,
        reference: SourceReference,
        *,
        now: float | None = None,
        max_age: float | None = None,
    ) -> bool:
        document_id = reference.id.split("#L", 1)[0]
        document = self.document(document_id)
        return (
            document is not None
            and reference.available
            and document.source.available
            and document.source.revision == reference.revision
            and (
                now is None
                or max_age is None
                or now
                - (
                    document.checked_at
                    if document.checked_at is not None
                    else document.source.observed_at
                )
                <= max_age
            )
            and all(
                status.availability == Availability.AVAILABLE
                for status in self.statuses()
                if status.provider == reference.provider
            )
        )

    def mark_removed(self, reference: SourceReference, checked_at: float) -> None:
        if self.document(reference.id) is None:
            return
        checkpoint = self.checkpoint()
        with transaction(self.db):
            self.db.execute(
                "UPDATE companion_documents SET active=0 WHERE pet=? AND provider=? AND source_id=?",
                (self.pet, reference.provider, reference.id),
            )
            self._change(replace(reference, observed_at=checked_at), "removed")
        self._notify(checkpoint)
