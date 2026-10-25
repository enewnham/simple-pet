from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field

from .commentary.request import SpeechConditions
from .domain import DeliveryStatus, SpeechReason
from .identity import SessionIdentity
from .logging import get_logger
from .runtime import Trigger

log = get_logger("pet-daemon")


class DeliveryConditions(BaseModel):
    commentary: SpeechConditions
    trigger: Trigger | None
    request_epoch: int


class SpeechPacket(TypedDict):
    text: str
    ttl: float
    delivery_id: str


class SpeechMessage(BaseModel):
    text: str
    ttl: float
    reason: SpeechReason
    selection_id: str
    subject_id: SessionIdentity | None = None
    revision: str | None = None
    history_epoch: int = 0
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conditions: DeliveryConditions | None = None
    mentioned_sources: tuple[str, ...] = ()


class DeliveryOutcome(BaseModel):
    message: SpeechMessage
    status: DeliveryStatus


@dataclass
class Attempt:
    message: SpeechMessage
    attempts: int = 0
    status: DeliveryStatus = DeliveryStatus.PENDING
    due: float = 0
    expires: float = 0


class SpeechDelivery:
    def __init__(
        self,
        emit: Callable[[SpeechPacket], object],
        *,
        clock: Callable[[], float] = time.monotonic,
        on_outcome: Callable[[DeliveryOutcome], None],
        valid: Callable[[SpeechMessage], bool],
        ack_timeout: float = 2,
    ) -> None:
        self.emit = emit
        self.clock = clock
        self.on_outcome = on_outcome
        self.valid = valid
        self.ack_timeout = ack_timeout
        self.deliveries: dict[str, Attempt] = {}

    def submit(self, message: SpeechMessage) -> str:
        delivery = self.deliveries.get(message.message_id)
        if delivery is None:
            delivery = Attempt(message)
            self.deliveries[message.message_id] = delivery
            self._send(delivery)
        return message.message_id

    def has_pending(self) -> bool:
        return any(
            delivery.status == DeliveryStatus.PENDING
            for delivery in self.deliveries.values()
        )

    def _send(self, delivery: Attempt) -> None:
        message = delivery.message
        if not self.valid(message):
            delivery.status = DeliveryStatus.CANCELLED
            return
        delivery.attempts += 1
        delivery.due = self.clock() + self.ack_timeout
        delivery.expires = self.clock() + message.ttl + 2 * self.ack_timeout
        try:
            self.emit(
                {
                    "text": message.text,
                    "ttl": message.ttl,
                    "delivery_id": message.message_id,
                }
            )
        except OSError:
            log.exception(
                "speech.send_failed",
                message=message.message_id,
                attempt=delivery.attempts,
            )

    def acknowledge(self, packet: Mapping[str, object]) -> bool:
        if packet.get("event") != "speech_accepted":
            return False
        identifier = packet.get("delivery_id")
        if not isinstance(identifier, str):
            return False
        delivery = self.deliveries.get(identifier)
        if (
            delivery is None
            or not delivery.attempts
            or delivery.status == DeliveryStatus.ACCEPTED
            or self.clock() > delivery.expires
        ):
            return False
        self.on_outcome(
            DeliveryOutcome(message=delivery.message, status=DeliveryStatus.ACCEPTED)
        )
        delivery.status = DeliveryStatus.ACCEPTED
        return True

    def tick(self) -> None:
        now = self.clock()
        for identifier, delivery in tuple(self.deliveries.items()):
            if delivery.status == DeliveryStatus.PENDING:
                if not self.valid(delivery.message):
                    delivery.status = DeliveryStatus.CANCELLED
                elif now >= delivery.due:
                    if delivery.attempts < 2:
                        self._send(delivery)
                    else:
                        self.on_outcome(
                            DeliveryOutcome(
                                message=delivery.message, status=DeliveryStatus.FAILED
                            )
                        )
                        delivery.status = DeliveryStatus.FAILED
            if delivery.status != DeliveryStatus.PENDING and now > delivery.expires:
                del self.deliveries[identifier]

    def clear(self) -> None:
        self.deliveries.clear()

    def cancel_pending(self) -> None:
        for delivery in self.deliveries.values():
            if delivery.status == DeliveryStatus.PENDING:
                delivery.status = DeliveryStatus.CANCELLED

    def close(self) -> None:
        self.clear()
