import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from .git import RunCommand, command
from .github import pull_request_identity


class ChecksStatus(StrEnum):
    PASSED = "passed"
    WAITING = "waiting"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChecksObservation:
    ref: str
    checked_at: float
    status: ChecksStatus
    head: str = ""
    revision: str = ""
    error: str | None = None


def check_status(record: dict[str, object]) -> ChecksStatus:
    state = record.get("state")
    if state is not None:
        if state == "SUCCESS":
            return ChecksStatus.PASSED
        if state in ("PENDING", "EXPECTED"):
            return ChecksStatus.WAITING
        return (
            ChecksStatus.FAILED
            if state in ("FAILURE", "ERROR")
            else ChecksStatus.UNKNOWN
        )
    if record.get("status") in (
        "QUEUED",
        "IN_PROGRESS",
        "WAITING",
        "PENDING",
        "REQUESTED",
    ):
        return ChecksStatus.WAITING
    if record.get("status") != "COMPLETED":
        return ChecksStatus.UNKNOWN
    conclusion = record.get("conclusion")
    if conclusion in ("SUCCESS", "NEUTRAL", "SKIPPED"):
        return ChecksStatus.PASSED
    if conclusion in (
        "FAILURE",
        "CANCELLED",
        "TIMED_OUT",
        "ACTION_REQUIRED",
        "STALE",
        "STARTUP_FAILURE",
    ):
        return ChecksStatus.FAILED
    return ChecksStatus.UNKNOWN


class GitHubChecks:
    def __init__(
        self, run: RunCommand = command, clock: Callable[[], float] = time.time
    ) -> None:
        self.run = run
        self.clock = clock

    async def observe(self, ref: str) -> ChecksObservation:
        repository, number = pull_request_identity(ref)
        if repository is None or number is None:
            raise ValueError("A watch requires a canonical pull request URL")
        try:
            output = await self.run(
                "gh", "pr", "view", ref, "--json", "headRefOid,statusCheckRollup"
            )
            data = json.loads(output)
            if not isinstance(data, dict):
                raise TypeError("GitHub returned an invalid checks response")
            head = data.get("headRefOid")
            checks = data.get("statusCheckRollup")
            if (
                not isinstance(head, str)
                or not head
                or not isinstance(checks, list)
                or not checks
            ):
                return ChecksObservation(
                    ref,
                    self.clock(),
                    ChecksStatus.UNKNOWN,
                    error="Current head or checks unavailable",
                )
            statuses = [
                check_status(record)
                if isinstance(record, dict)
                else ChecksStatus.UNKNOWN
                for record in checks
            ]
            if ChecksStatus.FAILED in statuses:
                status = ChecksStatus.FAILED
            elif ChecksStatus.UNKNOWN in statuses:
                status = ChecksStatus.UNKNOWN
            elif ChecksStatus.WAITING in statuses:
                status = ChecksStatus.WAITING
            else:
                status = ChecksStatus.PASSED
            revision = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            return ChecksObservation(ref, self.clock(), status, head, revision)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as error:
            return ChecksObservation(
                ref, self.clock(), ChecksStatus.UNKNOWN, error=str(error)
            )
