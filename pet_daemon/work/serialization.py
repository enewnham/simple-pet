from .models import (
    CheckoutAssociation,
    EvidencePayload,
    EvidenceRecord,
    PullRequestAssociation,
    StackAssociation,
)


def payload_record(payload: EvidencePayload) -> dict[str, object]:
    data = payload.model_dump(mode="json", exclude={"type", "source", "kind"})
    match payload:
        case CheckoutAssociation():
            return {
                **data,
                "relationship": "session_checkout",
                "association_only": True,
            }
        case PullRequestAssociation():
            return {
                **data,
                "relationship": "checkout_pull_request",
                "association_only": True,
            }
        case StackAssociation():
            return {
                **payload.link.model_dump(mode="json"),
                "from": payload.from_ref,
                "association_only": True,
            }
        case _:
            return data


def evidence_record(record: EvidenceRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "kind": record.kind,
        "data": payload_record(record.payload),
        "observed_at": record.observed_at,
    }
