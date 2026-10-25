import sqlite3
from contextlib import AbstractContextManager

from ..identity import SessionIdentity
from ..repository_identity import RepositoryIdentity, repository_columns
from ..work.association import TERMINAL_EVENTS
from ..work.history import report_prose
from ..work.models import (
    ExecutionEvidence,
    IntentEvidence,
    ProgressEvidence,
    SessionRecord,
    SessionReport,
    WorkFacts,
)
from ..work.observations import (
    BranchObservation,
    CheckoutObservation,
    PullRequestObservation,
    RepositoryObservation,
    SessionObservation,
    repository_observation_adapter,
)
from .identities import (
    optional_session_from_row,
    optional_session_values,
    session_from_row,
    session_parameters,
    session_values,
)
from .transactions import transaction

TABLES = (
    "work_sessions",
    "work_reports",
    "work_aliases",
    "work_checkouts",
    "work_branches",
    "work_pull_requests",
)


class SQLiteWorkRepository:
    def __init__(self, db: sqlite3.Connection, pet: str) -> None:
        self.db = db
        self.pet = pet

    def transaction(self) -> AbstractContextManager[None]:
        return transaction(self.db)

    def aliases(self) -> dict[SessionIdentity, SessionIdentity]:
        return {
            session_from_row(row, "alias_"): session_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM work_aliases WHERE pet=?", (self.pet,)
            )
        }

    def session_location(self, session: SessionIdentity) -> str:
        row = self.db.execute(
            "SELECT location FROM work_sessions WHERE pet=? AND source=? AND session=?",
            (self.pet, *session_values(session)),
        ).fetchone()
        return row["location"] if row else ""

    def save_session(self, observation: SessionObservation, now: float) -> None:
        identity = observation.session
        identities = (identity, *observation.aliases)
        placeholders, values = session_parameters(identities)
        records = self.db.execute(
            f"SELECT parent_source,parent_session,location,location_at FROM work_sessions WHERE pet=? AND (source,session) IN ({placeholders}) ORDER BY location_at DESC",
            (self.pet, *values),
        ).fetchall()
        location = next((row["location"] for row in records if row["location"]), "")
        location_at = next(
            (row["location_at"] for row in records if row["location"]), 0
        )
        if observation.location is not None and observation.location != location:
            location, location_at = observation.location, now
        parent = observation.parent or next(
            (
                optional_session_from_row(row, "parent_")
                for row in records
                if row["parent_source"] is not None
            ),
            None,
        )
        self.db.execute(
            "INSERT INTO work_sessions(pet,source,session,parent_source,parent_session,location,location_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(pet,source,session) DO UPDATE SET "
            "parent_source=excluded.parent_source,parent_session=excluded.parent_session,location=excluded.location,location_at=excluded.location_at",
            (
                self.pet,
                *session_values(identity),
                *optional_session_values(parent),
                location,
                location_at,
            ),
        )
        for alias in observation.aliases:
            if alias == identity:
                continue
            self.db.execute(
                "UPDATE work_aliases SET source=?,session=? WHERE pet=? AND source=? AND session=?",
                (*session_values(identity), self.pet, *session_values(alias)),
            )
            self.db.execute(
                "UPDATE work_sessions SET parent_source=?,parent_session=? WHERE pet=? AND parent_source=? AND parent_session=?",
                (*session_values(identity), self.pet, *session_values(alias)),
            )
            self.db.execute(
                "DELETE FROM work_sessions WHERE pet=? AND source=? AND session=?",
                (self.pet, *session_values(alias)),
            )
            self.bind_alias(alias, identity)

    def append_report(self, report: SessionReport) -> bool:
        session = report.payload.session
        kind = report.record().kind
        identities = (
            session,
            *(alias for alias, target in self.aliases().items() if target == session),
        )
        placeholders, values = session_parameters(identities)
        if isinstance(
            report.payload, ExecutionEvidence
        ) and report.payload.event in TERMINAL_EVENTS - {"session_end"}:
            rows = self.db.execute(
                f"SELECT data FROM work_reports WHERE pet=? AND (source,session) IN ({placeholders}) AND kind=? ORDER BY id DESC",
                (self.pet, *values, kind),
            )
            for row in rows:
                previous = SessionReport.model_validate_json(row["data"])
                payload = previous.payload
                if (
                    isinstance(payload, ExecutionEvidence)
                    and payload.event != "session_end"
                ):
                    if payload.event in TERMINAL_EVENTS:
                        return False
                    break
        row = self.db.execute(
            f"SELECT data FROM work_reports WHERE pet=? AND (source,session) IN ({placeholders}) AND kind=? ORDER BY id DESC LIMIT 1",
            (self.pet, *values, kind),
        ).fetchone()
        if row and not isinstance(report.payload, IntentEvidence):
            recorded = SessionReport.model_validate_json(row["data"])
            if (
                recorded.payload == report.payload
                and recorded.context == report.context
            ):
                return False
        cursor = self.db.execute(
            "INSERT INTO work_reports(pet,source,session,kind,data,observed_at) VALUES (?,?,?,?,?,?)",
            (
                self.pet,
                *session_values(session),
                kind,
                report.model_dump_json(),
                report.observed_at,
            ),
        )
        prose = report_prose(report)
        if prose:
            self.db.execute(
                "INSERT INTO work_reports_fts(rowid,prose) VALUES (?,?)",
                (cursor.lastrowid, prose),
            )
        return True

    def bind_alias(self, alias: SessionIdentity, session: SessionIdentity) -> None:
        self.db.execute(
            "INSERT INTO work_aliases VALUES (?,?,?,?,?) ON CONFLICT(pet,alias_source,alias_session) DO UPDATE SET source=excluded.source,session=excluded.session",
            (self.pet, *session_values(alias), *session_values(session)),
        )

    def get_checkout(
        self, path: str, *, enclosing: bool = False
    ) -> CheckoutObservation | None:
        value = (
            self._observation(
                "work_checkouts",
                "(worktree=? OR instr(?,rtrim(worktree,'/')||'/')=1) ORDER BY length(worktree) DESC LIMIT 1",
                (path, path),
            )
            if enclosing
            else self._observation("work_checkouts", "worktree=?", (path,))
        )
        assert value is None or isinstance(value, CheckoutObservation)
        return value

    def get_branch(
        self, repository: RepositoryIdentity, branch: str
    ) -> BranchObservation | None:
        value = self._observation(
            "work_branches",
            "repository_kind=? AND repository_value=? AND branch=?",
            (*repository_columns(repository), branch),
        )
        assert value is None or isinstance(value, BranchObservation)
        return value

    def get_pull_request(self, ref: str) -> PullRequestObservation | None:
        value = self._observation("work_pull_requests", "ref=?", (ref,))
        assert value is None or isinstance(value, PullRequestObservation)
        return value

    def _observation(
        self, table: str, where: str, values: tuple[str | None, ...]
    ) -> RepositoryObservation | None:
        row = self.db.execute(
            f"SELECT data FROM {table} WHERE pet=? AND {where}", (self.pet, *values)
        ).fetchone()
        return (
            repository_observation_adapter.validate_json(row["data"]) if row else None
        )

    def put_observation(self, observation: RepositoryObservation) -> None:
        data = observation.model_dump_json()
        match observation:
            case CheckoutObservation():
                self.db.execute(
                    "INSERT INTO work_checkouts VALUES (?,?,?) ON CONFLICT(pet,worktree) DO UPDATE SET data=excluded.data",
                    (self.pet, observation.worktree, data),
                )
            case BranchObservation():
                self.db.execute(
                    "INSERT INTO work_branches VALUES (?,?,?,?,?) ON CONFLICT(pet,repository_kind,repository_value,branch) DO UPDATE SET data=excluded.data",
                    (
                        self.pet,
                        *repository_columns(observation.repository),
                        observation.branch,
                        data,
                    ),
                )
            case PullRequestObservation():
                self.db.execute(
                    "INSERT INTO work_pull_requests VALUES (?,?,?) ON CONFLICT(pet,ref) DO UPDATE SET data=excluded.data",
                    (self.pet, observation.ref, data),
                )

    def _session(self, row: sqlite3.Row) -> SessionRecord:
        identity = session_from_row(row)
        identities = (
            identity,
            *(alias for alias, target in self.aliases().items() if target == identity),
        )
        placeholders, values = session_parameters(identities)
        records = self.db.execute(
            f"SELECT data FROM work_reports WHERE pet=? AND (source,session) IN ({placeholders}) ORDER BY id",
            (self.pet, *values),
        )
        reports = tuple(
            SessionReport.model_validate_json(record["data"]) for record in records
        )
        authored = tuple(
            report
            for report in reports
            if isinstance(report.payload, IntentEvidence | ProgressEvidence)
        )
        execution = next(
            (
                report
                for report in reversed(reports)
                if isinstance(report.payload, ExecutionEvidence)
                and report.payload.event != "session_end"
            ),
            None,
        )
        intent = next(
            (
                report.payload.text
                for report in reversed(authored)
                if isinstance(report.payload, IntentEvidence)
            ),
            "",
        )
        return SessionRecord(
            id=identity,
            goal=intent,
            parent=optional_session_from_row(row, "parent_"),
            location=row["location"],
            location_at=row["location_at"],
            reports=authored,
            execution=execution,
            closure=next(
                (
                    report
                    for report in reversed(reports)
                    if isinstance(report.payload, ExecutionEvidence)
                    and report.payload.event == "session_end"
                ),
                None,
            ),
            changed_at=max(
                max((report.observed_at for report in authored), default=0),
                execution.observed_at if execution else 0,
            ),
            aliases=identities[1:],
        )

    def load(self) -> WorkFacts:
        state = WorkFacts()
        state.aliases = {
            session_from_row(row, "alias_"): session_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM work_aliases WHERE pet=?", (self.pet,)
            )
        }
        state.sessions = {
            session_from_row(row): self._session(row)
            for row in self.db.execute(
                "SELECT * FROM work_sessions WHERE pet=?", (self.pet,)
            )
        }
        for table in ("work_checkouts", "work_branches", "work_pull_requests"):
            for row in self.db.execute(
                f"SELECT data FROM {table} WHERE pet=?", (self.pet,)
            ):
                value = repository_observation_adapter.validate_json(row["data"])
                match value:
                    case CheckoutObservation():
                        state.checkouts[value.worktree] = value
                    case BranchObservation():
                        state.branches[value.repository, value.branch] = value
                    case PullRequestObservation():
                        state.pull_requests[value.ref] = value
        return state

    def clear(self) -> None:
        for table in TABLES:
            self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))
