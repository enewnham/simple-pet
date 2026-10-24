import unittest

from pydantic import ValidationError

from pet_daemon.domain import (
    ArtifactKind,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    PullRequestState,
    SessionSource,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.repository_identity import LocalRepository, RemoteRepository
from pet_daemon.work.models import (
    ArtifactReference,
    AvailabilityEvidence,
    CheckoutAssociation,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
    ReferenceAssociation,
    RelatedLink,
    ReportContext,
    SessionReport,
    StackAssociation,
)
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
    SessionObservation,
    repository_observation_adapter,
)


class ModelSerializationTests(unittest.TestCase):
    def test_evidence_variants_retain_nested_identity_and_type(self) -> None:
        session = SessionIdentity(source=SessionSource.CODEX, native_id="session")
        repository = RemoteRepository(value="host/org/repo")
        link = RelatedLink(
            kind=ArtifactKind.PULL_REQUEST,
            ref="https://host/org/repo/pull/1",
            relationship=LinkRelationship.STACK_ENTRY,
            source=EvidenceSource.GITHUB_STACK,
            is_draft=False,
        )
        pr = PullRequest(
            number=1,
            url=link.ref,
            state=PullRequestState.OPEN,
            merged_at=None,
            repository=repository,
            related_links=(link,),
        )
        variants = (
            IntentEvidence(session=session, text=""),
            ProgressEvidence(session=session, text="progress"),
            ExecutionEvidence(
                session=session,
                event="end",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress="",
            ),
            CheckoutAssociation(checkout="/checkout", branch="main"),
            PullRequestAssociation(checkout="/checkout", branch="main"),
            ReferenceAssociation(ref="ticket", source=EvidenceSource.AGENT_REFERENCE),
            GitHubEvidence(pr=None, repository=RemoteRepository(value="host/org/repo")),
            AvailabilityEvidence(local_error="", github_error=None),
            StackAssociation(link=link, from_ref="https://host/org/repo/pull/2"),
            GitHubEvidence(pr=pr, repository=repository),
            GitObservation(
                worktree="/checkout",
                branch="main",
                head="abc",
                staged=0,
                unstaged=0,
                untracked=0,
                conflicts=0,
                upstream="",
                ahead=0,
                behind=0,
                repository=repository,
            ),
        )
        for payload in variants:
            with self.subTest(variant=type(payload).__name__):
                record = EvidenceRecord(payload=payload, observed_at=1.5)
                decoded = EvidenceRecord.model_validate_json(record.model_dump_json())
                self.assertEqual(decoded, record)
                self.assertIs(type(decoded.payload), type(payload))

    def test_repository_observation_variants_retain_empty_and_false(self) -> None:
        repository = LocalRepository(path="/checkout")
        local = GitObservation(
            worktree="/checkout",
            branch="",
            head="",
            staged=0,
            unstaged=0,
            untracked=0,
            conflicts=0,
            upstream="",
            ahead=0,
            behind=0,
            repository=repository,
        )
        observations = (
            CheckoutObservation(
                worktree="/checkout", local=local, checked_at=2, available=False
            ),
            BranchObservation(repository=repository, branch="", pr=None, checked_at=2),
            PullRequestObservation(
                ref="https://host/org/repo/pull/1", pr=None, checked_at=2
            ),
        )
        for observation in observations:
            with self.subTest(variant=type(observation).__name__):
                decoded = repository_observation_adapter.validate_json(
                    observation.model_dump_json()
                )
                self.assertEqual(decoded, observation)
                self.assertIs(type(decoded), type(observation))

    def test_report_round_trip_retains_empty_text_and_nested_repository(self) -> None:
        report = SessionReport(
            payload=IntentEvidence(
                session=SessionIdentity(
                    source=SessionSource.CLAUDE, native_id="report"
                ),
                text="",
            ),
            context=ReportContext(
                repository=RemoteRepository(value="host/org/repo"),
                references=(
                    ArtifactReference(
                        kind=ArtifactKind.BRANCH,
                        ref="main",
                        repository=LocalRepository(path="/checkout"),
                    ),
                ),
            ),
            observed_at=3,
        )
        self.assertEqual(
            SessionReport.model_validate_json(report.model_dump_json()), report
        )

    def test_session_observation_decodes_each_report_discriminator(self) -> None:
        session = SessionIdentity(source=SessionSource.CODEX, native_id="observation")
        reports = (
            IntentEvidence(session=session, text="intent"),
            ProgressEvidence(session=session, text="progress"),
            ExecutionEvidence(
                session=session,
                event="end",
                outcome=None,
                tool_name=None,
                progress=None,
            ),
        )
        observation = SessionObservation(session=session, reports=reports)
        restored = SessionObservation.model_validate_json(observation.model_dump_json())
        self.assertEqual(restored, observation)
        self.assertEqual(
            tuple(type(report) for report in restored.reports),
            tuple(type(report) for report in reports),
        )
        invalid = observation.model_dump(mode="json")
        invalid["reports"][0]["type"] = "unknown"
        with self.assertRaises(ValidationError):
            SessionObservation.model_validate(invalid)

    def test_invalid_union_discriminator_and_identity_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceRecord.model_validate({"payload": {"type": "unknown"}})
        with self.assertRaises(ValidationError):
            SessionIdentity.model_validate(
                {"source": "unknown", "native_id": "session"}
            )
        with self.assertRaises(ValidationError):
            repository_observation_adapter.validate_python({"kind": "unknown"})
