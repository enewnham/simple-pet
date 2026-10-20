from .models import Opportunity
from .views import EvidenceView, WorkView, project_evidence


def evidence_record(record: EvidenceView):
    return {"source": record.source, "kind": record.kind, **record.details,
            **({"observed_at": record.observed_at} if record.observed_at is not None else {})}


def work_record(work: WorkView):
    return {"intent": work.intent, "requests": tuple(evidence_record(record) for record in work.requests),
            "activity": tuple(evidence_record(record) for record in work.activity),
            "artifacts": tuple({"kind": item.kind, "ref": item.ref, **({"title": item.title} if item.title else {}),
                                "evidence": tuple(evidence_record(record) for record in item.evidence)} for item in work.artifacts)}


def opportunity_record(action: Opportunity):
    artifact = action.artifact
    return {"kind": action.kind, "intent": action.intent[:1000],
            "artifact": {key: artifact[key][:240] if key == "title" else artifact[key] for key in ("kind", "ref", "title") if key in artifact},
            "evidence": tuple(evidence_record(value) for record in action.evidence if (value := project_evidence(record)) is not None),
            "uncertainty": action.uncertainty, "next_step": action.next_step}
