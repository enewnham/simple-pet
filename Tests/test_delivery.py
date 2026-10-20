import unittest
import uuid

from pet_daemon.delivery import SpeechDelivery


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 10
        self.sent = []
        self.outcomes = []
        self.current = True

    def delivery(self, **options):
        return SpeechDelivery(self.sent.append, clock=lambda: self.now,
                              on_outcome=lambda *args: self.outcomes.append(args),
                              valid=lambda *args: self.current, **options)

    def test_only_matching_acceptance_surfaces_the_opportunity(self):
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        uuid.UUID(identifier)
        self.assertFalse(delivery.can_submit("work", 1))
        self.assertTrue(delivery.can_submit("work", 2))
        self.assertEqual(delivery.speech("work", 1), "Tests passed")
        self.assertEqual(self.sent, [{"text": "Tests passed", "ttl": 10, "delivery_id": identifier}])
        for packet in ({"event": "animation", "delivery_id": identifier},
                       {"event": "speech_accepted", "delivery_id": str(uuid.uuid4())},
                       {"event": "speech_accepted", "command_id": identifier}):
            self.assertFalse(delivery.acknowledge(packet))
        self.assertEqual(self.outcomes, [])
        packet = {"event": "speech_accepted", "delivery_id": identifier}
        self.assertTrue(delivery.acknowledge(packet))
        self.assertFalse(delivery.acknowledge(packet))
        self.assertEqual(self.outcomes, [("work", 1, "accepted")])
        self.now += 20
        delivery.tick()
        self.assertEqual(len(self.sent), 1)

    def test_missing_acknowledgment_retries_once_and_unchanged_polling_stays_exhausted(self):
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        self.now += 2
        delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.sent[0], self.sent[1])
        self.now += 2
        delivery.tick()
        self.assertEqual(self.outcomes, [("work", 1, "failed")])
        self.assertEqual(delivery.submit("Changed text", 10, "work", 1), identifier)
        self.now += 20
        delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertFalse(delivery.can_submit("work", 1))

    def test_send_failures_use_the_same_bounded_retry_budget(self):
        def unavailable(packet):
            self.sent.append(packet)
            raise OSError("offline")
        delivery = self.delivery()
        delivery.emit = unavailable
        delivery.submit("Tests passed", 10, "work", 1)
        self.now += 2
        delivery.tick()
        self.now += 2
        delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.outcomes, [("work", 1, "failed")])

    def test_superseded_opportunities_neither_retry_nor_accept(self):
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        self.current = False
        self.assertFalse(delivery.acknowledge({"event": "speech_accepted", "delivery_id": identifier}))
        self.now += 10
        delivery.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.outcomes, [])

    def test_cancellation_waits_for_resubmission_and_preserves_the_original_packet(self):
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        delivery.cancel_pending()
        self.now += 60
        delivery.tick()
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(delivery.acknowledge({"event": "speech_accepted", "delivery_id": identifier}))
        self.assertTrue(delivery.can_submit("work", 1))
        self.assertEqual(delivery.submit("Regenerated text", 99, "work", 1), identifier)
        self.assertEqual(self.sent, [{"text": "Tests passed", "ttl": 10, "delivery_id": identifier}] * 2)
        delivery.cancel_pending()
        self.assertFalse(delivery.can_submit("work", 1))
        self.assertEqual(delivery.submit("Another attempt", 10, "work", 1), identifier)
        self.now += 60
        delivery.tick()
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.outcomes, [])

    def test_initially_ineligible_delivery_can_resume_without_consuming_a_send(self):
        self.current = False
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        self.assertEqual(self.sent, [])
        self.assertEqual(delivery.submit("Changed text", 20, "work", 1), identifier)
        self.assertEqual(self.sent, [])
        self.current = True
        self.assertTrue(delivery.can_submit("work", 1))
        self.assertEqual(delivery.submit("Regenerated text", 99, "work", 1), identifier)
        self.now += 2
        delivery.tick()
        self.assertEqual(self.sent, [{"text": "Tests passed", "ttl": 10, "delivery_id": identifier}] * 2)
        self.assertTrue(delivery.acknowledge({"event": "speech_accepted", "delivery_id": identifier}))
        self.assertEqual(self.outcomes, [("work", 1, "accepted")])

    def test_clearing_or_restarting_allows_unsurfaced_work_to_receive_a_fresh_attempt(self):
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        delivery.clear()
        self.assertTrue(delivery.can_submit("work", 1))
        self.assertIsNone(delivery.speech("work", 1))
        self.assertFalse(delivery.acknowledge({"event": "speech_accepted", "delivery_id": identifier}))
        second = delivery.submit("Fresh text", 10, "work", 1)
        self.assertNotEqual(second, identifier)
        delivery.close()
        self.now += 60
        delivery.tick()
        self.assertEqual(len(self.sent), 2)
        restarted = self.delivery()
        self.assertTrue(restarted.can_submit("work", 1))
        self.assertNotEqual(restarted.submit("Fresh text", 10, "work", 1), second)

    def test_failed_outcome_callback_leaves_acceptance_retryable(self):
        delivery = self.delivery()
        identifier = delivery.submit("Tests passed", 10, "work", 1)
        def unavailable(*args):
            raise RuntimeError("unavailable")
        delivery.on_outcome = unavailable
        packet = {"event": "speech_accepted", "delivery_id": identifier}
        with self.assertRaises(RuntimeError):
            delivery.acknowledge(packet)
        delivery.on_outcome = lambda *args: self.outcomes.append(args)
        self.assertTrue(delivery.acknowledge(packet))
        self.assertEqual(self.outcomes, [("work", 1, "accepted")])


if __name__ == "__main__":
    unittest.main()
