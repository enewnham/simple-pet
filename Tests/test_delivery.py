import unittest
import uuid
from typing import Never

from pet_daemon.delivery import (
    DeliveryOutcome,
    SpeechDelivery,
    SpeechMessage,
    SpeechPacket,
)
from pet_daemon.domain import SessionSource, SpeechReason
from pet_daemon.identity import SessionIdentity


class DeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 10
        self.sent: list[SpeechPacket] = []
        self.outcomes: list[DeliveryOutcome] = []
        self.current = True
        self.delivery = SpeechDelivery(
            self.sent.append,
            clock=lambda: self.now,
            on_outcome=self.outcomes.append,
            valid=lambda message: self.current,
        )
        self.message = SpeechMessage(
            text="The parser preserves quoted fields.",
            ttl=10,
            reason=SpeechReason.ROUTINE,
            selection_id="selection",
            subject_id=SessionIdentity(source=SessionSource.COMMAND, native_id="work"),
            revision="one",
        )

    def acknowledge(self, identifier: str | None = None) -> bool:
        return self.delivery.acknowledge(
            {
                "event": "speech_accepted",
                "delivery_id": identifier or self.message.message_id,
            }
        )

    def test_acceptance_matches_a_sent_message_once(self) -> None:
        identifier = self.delivery.submit(self.message)
        uuid.UUID(identifier)
        self.assertEqual(
            self.sent,
            [{"text": self.message.text, "ttl": 10, "delivery_id": identifier}],
        )
        self.assertFalse(self.acknowledge(str(uuid.uuid4())))
        self.assertFalse(
            self.delivery.acknowledge({"event": "animation", "delivery_id": identifier})
        )
        self.assertTrue(self.acknowledge())
        self.assertFalse(self.acknowledge())
        self.assertEqual(
            [
                (item.message.subject_id, item.message.revision, item.status)
                for item in self.outcomes
            ],
            [
                (
                    SessionIdentity(source=SessionSource.COMMAND, native_id="work"),
                    "one",
                    "accepted",
                )
            ],
        )

    def test_missing_acknowledgment_retries_the_same_message_once(self) -> None:
        self.delivery.submit(self.message)
        self.now += 2
        self.delivery.tick()
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.sent[0], self.sent[1])
        self.assertEqual([item.status for item in self.outcomes], ["failed"])
        self.delivery.submit(self.message)
        self.assertEqual(len(self.sent), 2)

    def test_send_errors_share_the_bounded_retry_budget(self) -> None:
        def unavailable(packet: SpeechPacket) -> Never:
            self.sent.append(packet)
            raise OSError("offline")

        self.delivery.emit = unavailable
        self.delivery.submit(self.message)
        for _ in range(3):
            self.now += 2
            self.delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual([item.status for item in self.outcomes], ["failed"])

    def test_changed_facts_stop_retries_but_acceptance_records_display(self) -> None:
        self.delivery.submit(self.message)
        self.current = False
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.acknowledge())
        self.assertEqual(self.outcomes[0].message.revision, "one")

    def test_preemption_stops_retries_and_retains_late_acceptance(self) -> None:
        self.delivery.submit(self.message)
        self.delivery.cancel_pending()
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.acknowledge())

    def test_late_acceptance_supersedes_transport_timeout(self) -> None:
        self.delivery.submit(self.message)
        for _ in range(2):
            self.now += 2
            self.delivery.tick()
        self.assertTrue(self.acknowledge())
        self.assertEqual(
            [item.status for item in self.outcomes], ["failed", "accepted"]
        )

    def test_unsent_invalid_message_cannot_be_acknowledged(self) -> None:
        self.current = False
        self.delivery.submit(self.message)
        self.assertEqual(self.sent, [])
        self.assertFalse(self.acknowledge())

    def test_clear_rejects_old_acceptance(self) -> None:
        self.delivery.submit(self.message)
        self.delivery.clear()
        self.assertFalse(self.acknowledge())
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 1)

    def test_sent_metadata_expires_after_display_and_acknowledgment_window(
        self,
    ) -> None:
        self.delivery.submit(self.message)
        self.delivery.cancel_pending()
        self.now += 15
        self.delivery.tick()
        self.assertEqual(self.delivery.deliveries, {})
        self.assertFalse(self.acknowledge())

    def test_persistence_failure_leaves_acceptance_retryable(self) -> None:
        self.delivery.submit(self.message)

        def unavailable(outcome: DeliveryOutcome) -> Never:
            raise RuntimeError("storage unavailable")

        self.delivery.on_outcome = unavailable
        with self.assertRaises(RuntimeError):
            self.acknowledge()
        self.delivery.on_outcome = self.outcomes.append
        self.assertTrue(self.acknowledge())
        self.assertEqual([item.status for item in self.outcomes], ["accepted"])


if __name__ == "__main__":
    unittest.main()
