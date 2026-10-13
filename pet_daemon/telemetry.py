import contextlib
import contextvars
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import uuid
import time


log = logging.getLogger('pet-daemon')
observation_id = contextvars.ContextVar('observation_id', default=None)


class JSONFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.run_id = uuid.uuid4().hex

    def format(self, record):
        entry = {'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                 'level': record.levelname, 'run_id': self.run_id,
                 'event': getattr(record, 'event', 'log'),
                 **getattr(record, 'fields', {})}
        if observation_id.get() is not None:
            entry['observation_id'] = observation_id.get()
        if not hasattr(record, 'event'):
            entry['message'] = record.getMessage()
        if record.exc_info:
            entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure(path, verbose=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8')
    file.setFormatter(JSONFormatter())
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter('pet-daemon: %(message)s'))
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.addHandler(file)
    log.addHandler(console)
    log.info('logging to %s', path)


def trace(event, **fields):
    log.debug(event, extra={'event': event, 'fields': {'monotonic_seconds': time.monotonic(), **fields}})


@contextlib.contextmanager
def observation():
    token = observation_id.set(uuid.uuid4().hex)
    try:
        yield
    finally:
        observation_id.reset(token)
