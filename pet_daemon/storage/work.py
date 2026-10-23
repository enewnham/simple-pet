import json
import sqlite3
from contextlib import AbstractContextManager

from ..work.association import TERMINAL_EVENTS
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
    repository_observation_from_mapping,
)
from ..work.serialization import report_data, report_from_data
from ..work.validation import canonical
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

    def aliases(self) -> dict[str, str]:
        return {
            row["alias"]: row["session"]
            for row in self.db.execute(
                "SELECT alias,session FROM work_aliases WHERE pet=?", (self.pet,)
            )
        }

    def session_location(self, session: str) -> str:
        row = self.db.execute(
            "SELECT location FROM work_sessions WHERE pet=? AND session=?",
            (self.pet, session),
        ).fetchone()
        return row["location"] if row else ""

    def save_session(self, observation: SessionObservation, now: float) -> None:
        identity = observation.session
        identities = (identity, *observation.aliases)
        placeholders = ",".join("?" for _ in identities)
        records = self.db.execute(
            f"SELECT parent,location,location_at FROM work_sessions WHERE pet=? AND session IN ({placeholders}) ORDER BY location_at DESC",
            (self.pet, *identities),
        ).fetchall()
        location = next((row["location"] for row in records if row["location"]), "")
        location_at = next(
            (row["location_at"] for row in records if row["location"]), 0
        )
        if observation.location is not None and observation.location != location:
            location, location_at = observation.location, now
        parent = observation.parent or next(
            (row["parent"] for row in records if row["parent"]), None
        )
        self.db.execute(
            "INSERT INTO work_sessions(pet,session,parent,location,location_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(pet,session) DO UPDATE SET "
            "parent=excluded.parent,location=excluded.location,location_at=excluded.location_at",
            (self.pet, identity, parent, location, location_at),
        )
        for alias in observation.aliases:
            if alias == identity:
                continue
            self.db.execute(
                "UPDATE work_aliases SET session=? WHERE pet=? AND session=?",
                (identity, self.pet, alias),
            )
            self.db.execute(
                "UPDATE work_sessions SET parent=? WHERE pet=? AND parent=?",
                (identity, self.pet, alias),
            )
            self.db.execute(
                "DELETE FROM work_sessions WHERE pet=? AND session=?",
                (self.pet, alias),
            )
            self.bind_alias(alias, identity)

    def append_report(self, report: SessionReport) -> bool:
        session = report.payload.session
        kind = report.record().kind
        identities = (
            session,
            *(alias for alias, target in self.aliases().items() if target == session),
        )
        placeholders = ",".join("?" for _ in identities)
        if isinstance(
            report.payload, ExecutionEvidence
        ) and report.payload.event in TERMINAL_EVENTS - {"session_end"}:
            rows = self.db.execute(
                f"SELECT data FROM work_reports WHERE pet=? AND session IN ({placeholders}) AND kind=? ORDER BY id DESC",
                (self.pet, *identities, kind),
            )
            for row in rows:
                previous = report_from_data(json.loads(row["data"]))
                payload = previous.payload
                if (
                    isinstance(payload, ExecutionEvidence)
                    and payload.event != "session_end"
                ):
                    if payload.event in TERMINAL_EVENTS:
                        return False
                    break
        row = self.db.execute(
            f"SELECT data FROM work_reports WHERE pet=? AND session IN ({placeholders}) AND kind=? ORDER BY id DESC LIMIT 1",
            (self.pet, *identities, kind),
        ).fetchone()
        if row and not isinstance(report.payload, IntentEvidence):
            recorded = report_from_data(json.loads(row["data"]))
            if (
                recorded.payload == report.payload
                and recorded.context == report.context
            ):
                return False
        self.db.execute(
            "INSERT INTO work_reports(pet,session,kind,data,observed_at) VALUES (?,?,?,?,?)",
            (
                self.pet,
                session,
                kind,
                canonical(report_data(report)),
                report.observed_at,
            ),
        )
        return True

    def bind_alias(self, alias: str, session: str) -> None:
        self.db.execute(
            "INSERT INTO work_aliases VALUES (?,?,?) ON CONFLICT(pet,alias) DO UPDATE SET session=excluded.session",
            (self.pet, alias, session),
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

    def get_branch(self, repository: str, branch: str) -> BranchObservation | None:
        value = self._observation(
            "work_branches", "repository=? AND branch=?", (repository, branch)
        )
        assert value is None or isinstance(value, BranchObservation)
        return value

    def get_pull_request(self, ref: str) -> PullRequestObservation | None:
        value = self._observation("work_pull_requests", "ref=?", (ref,))
        assert value is None or isinstance(value, PullRequestObservation)
        return value

    def _observation(
        self, table: str, where: str, values: tuple[str, ...]
    ) -> RepositoryObservation | None:
        row = self.db.execute(
            f"SELECT data FROM {table} WHERE pet=? AND {where}", (self.pet, *values)
        ).fetchone()
        return (
            repository_observation_from_mapping(json.loads(row["data"]))
            if row
            else None
        )

    def put_observation(self, observation: RepositoryObservation) -> None:
        data = canonical(observation.packet())
        match observation:
            case CheckoutObservation():
                self.db.execute(
                    "INSERT INTO work_checkouts VALUES (?,?,?) ON CONFLICT(pet,worktree) DO UPDATE SET data=excluded.data",
                    (self.pet, observation.worktree, data),
                )
            case BranchObservation():
                self.db.execute(
                    "INSERT INTO work_branches VALUES (?,?,?,?) ON CONFLICT(pet,repository,branch) DO UPDATE SET data=excluded.data",
                    (self.pet, observation.repository, observation.branch, data),
                )
            case PullRequestObservation():
                self.db.execute(
                    "INSERT INTO work_pull_requests VALUES (?,?,?) ON CONFLICT(pet,ref) DO UPDATE SET data=excluded.data",
                    (self.pet, observation.ref, data),
                )

    def _session(self, row: sqlite3.Row) -> SessionRecord:
        identity = row["session"]
        identities = (
            identity,
            *(alias for alias, target in self.aliases().items() if target == identity),
        )
        placeholders = ",".join("?" for _ in identities)
        records = self.db.execute(
            f"SELECT data FROM work_reports WHERE pet=? AND session IN ({placeholders}) ORDER BY id",
            (self.pet, *identities),
        )
        reports = tuple(
            report_from_data(json.loads(record["data"])) for record in records
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
            parent=row["parent"],
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
            row["alias"]: row["session"]
            for row in self.db.execute(
                "SELECT * FROM work_aliases WHERE pet=?", (self.pet,)
            )
        }
        state.sessions = {
            row["session"]: self._session(row)
            for row in self.db.execute(
                "SELECT * FROM work_sessions WHERE pet=?", (self.pet,)
            )
        }
        for table in ("work_checkouts", "work_branches", "work_pull_requests"):
            for row in self.db.execute(
                f"SELECT data FROM {table} WHERE pet=?", (self.pet,)
            ):
                value = repository_observation_from_mapping(json.loads(row["data"]))
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
