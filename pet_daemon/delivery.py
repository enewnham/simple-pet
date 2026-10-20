from dataclasses import dataclass
import time
import uuid


@dataclass
class Attempt:
    delivery_id: str
    opportunity_id: str
    version: int
    text: str
    ttl: float
    attempts: int = 0
    status: str = "pending"
    due: float = 0


class SpeechDelivery:
    def __init__(self, emit, clock=time.monotonic, on_outcome=None, valid=None, ack_timeout=2):
        self.emit = emit
        self.clock = clock
        self.on_outcome = on_outcome or (lambda identifier, version, outcome: None)
        self.valid = valid or (lambda identifier, version: True)
        self.ack_timeout = ack_timeout
        self.deliveries = {}

    def submit(self, text, ttl, opportunity_id, version):
        key = (opportunity_id, version)
        delivery = self.deliveries.get(key)
        if delivery is not None:
            if delivery.status == "cancelled" and delivery.attempts < 2:
                delivery.status = "pending"
                self._send(delivery)
            return delivery.delivery_id
        delivery = Attempt(str(uuid.uuid4()), opportunity_id, version, text, ttl)
        self.deliveries[key] = delivery
        self._send(delivery)
        return delivery.delivery_id

    def _send(self, delivery):
        if not self.valid(delivery.opportunity_id, delivery.version):
            delivery.status = "cancelled"
            return
        delivery.attempts += 1
        delivery.due = self.clock() + self.ack_timeout
        try:
            self.emit({"text": delivery.text, "ttl": delivery.ttl, "delivery_id": delivery.delivery_id})
        except OSError:
            pass

    def acknowledge(self, packet):
        if packet.get("event") != "speech_accepted":
            return False
        delivery_id = packet.get("delivery_id")
        if not isinstance(delivery_id, str):
            return False
        delivery = next((item for item in self.deliveries.values() if item.delivery_id == delivery_id), None)
        if delivery is None or delivery.status != "pending":
            return False
        if not self.valid(delivery.opportunity_id, delivery.version):
            delivery.status = "cancelled"
            return False
        self.on_outcome(delivery.opportunity_id, delivery.version, "accepted")
        delivery.status = "accepted"
        return True

    def tick(self):
        for delivery in list(self.deliveries.values()):
            if delivery.status != "pending":
                continue
            if not self.valid(delivery.opportunity_id, delivery.version):
                delivery.status = "cancelled"
            elif self.clock() >= delivery.due:
                if delivery.attempts < 2:
                    self._send(delivery)
                else:
                    delivery.status = "failed"
                    self.on_outcome(delivery.opportunity_id, delivery.version, "failed")

    def can_submit(self, opportunity_id, version):
        delivery = self.deliveries.get((opportunity_id, version))
        return delivery is None or (delivery.status == "cancelled" and delivery.attempts < 2)

    def speech(self, opportunity_id, version):
        delivery = self.deliveries.get((opportunity_id, version))
        return delivery.text if delivery else None

    def clear(self):
        self.deliveries.clear()

    def cancel_pending(self):
        for delivery in self.deliveries.values():
            if delivery.status == "pending":
                delivery.status = "cancelled"

    def close(self):
        self.clear()
