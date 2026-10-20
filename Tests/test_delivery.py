import unittest
import uuid

from pet_daemon.delivery import SpeechDelivery, SpeechMessage


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 10
        self.sent = []
        self.outcomes = []
        self.current = True
        self.delivery = SpeechDelivery(self.sent.append, clock=lambda: self.now,
                                       on_outcome=self.outcomes.append,
                                       valid=lambda message: self.current)
        self.message = SpeechMessage("The PR requests your review.", 10, "routine", "selection",
                                     "work", 1, ("opportunity", 1))

    def acknowledge(self, identifier=None):
        return self.delivery.acknowledge({"event": "speech_accepted",
                                          "delivery_id": identifier or self.message.message_id})

    def test_acceptance_matches_a_sent_message_once(self):
        identifier = self.delivery.submit(self.message)
        uuid.UUID(identifier)
        self.assertEqual(self.sent, [{"text": self.message.text, "ttl": 10, "delivery_id": identifier}])
        self.assertFalse(self.acknowledge(str(uuid.uuid4())))
        self.assertFalse(self.delivery.acknowledge({"event": "animation", "delivery_id": identifier}))
        self.assertTrue(self.acknowledge())
        self.assertFalse(self.acknowledge())
        self.assertEqual([(item.message.opportunity, item.status) for item in self.outcomes],
                         [(("opportunity", 1), "accepted")])

    def test_missing_acknowledgment_retries_the_same_message_once(self):
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

    def test_send_errors_share_the_bounded_retry_budget(self):
        def unavailable(packet):
            self.sent.append(packet)
            raise OSError("offline")
        self.delivery.emit = unavailable
        self.delivery.submit(self.message)
        for _ in range(3):
            self.now += 2
            self.delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual([item.status for item in self.outcomes], ["failed"])

    def test_changed_facts_stop_retries_but_acceptance_records_display(self):
        self.delivery.submit(self.message)
        self.current = False
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.acknowledge())
        self.assertEqual(self.outcomes[0].message.opportunity, ("opportunity", 1))

    def test_preemption_stops_retries_and_retains_late_acceptance(self):
        self.delivery.submit(self.message)
        self.delivery.cancel_pending()
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.acknowledge())

    def test_late_acceptance_supersedes_transport_timeout(self):
        self.delivery.submit(self.message)
        for _ in range(2):
            self.now += 2
            self.delivery.tick()
        self.assertTrue(self.acknowledge())
        self.assertEqual([item.status for item in self.outcomes], ["failed", "accepted"])

    def test_unsent_invalid_message_cannot_be_acknowledged(self):
        self.current = False
        self.delivery.submit(self.message)
        self.assertEqual(self.sent, [])
        self.assertFalse(self.acknowledge())

    def test_clear_rejects_old_acceptance(self):
        self.delivery.submit(self.message)
        self.delivery.clear()
        self.assertFalse(self.acknowledge())
        self.now += 2
        self.delivery.tick()
        self.assertEqual(len(self.sent), 1)

    def test_sent_metadata_expires_after_display_and_acknowledgment_window(self):
        self.delivery.submit(self.message)
        self.delivery.cancel_pending()
        self.now += 15
        self.delivery.tick()
        self.assertEqual(self.delivery.deliveries, {})
        self.assertFalse(self.acknowledge())

    def test_persistence_failure_leaves_acceptance_retryable(self):
        self.delivery.submit(self.message)
        def unavailable(outcome):
            raise RuntimeError("storage unavailable")
        self.delivery.on_outcome = unavailable
        with self.assertRaises(RuntimeError):
            self.acknowledge()
        self.delivery.on_outcome = self.outcomes.append
        self.assertTrue(self.acknowledge())
        self.assertEqual([item.status for item in self.outcomes], ["accepted"])


if __name__ == "__main__":
    unittest.main()
