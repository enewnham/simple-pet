import contextlib
import contextvars
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import uuid
import time


log = logging.getLogger('pet-daemon')
VERBOSE = 5
logging.addLevelName(VERBOSE, 'VERBOSE')
observation_id = contextvars.ContextVar('observation_id', default=None)
request_usage = contextvars.ContextVar('request_usage', default=None)


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


class ConsoleFormatter(logging.Formatter):
    COLORS = {VERBOSE: '90', logging.DEBUG: '90', logging.INFO: '36', logging.WARNING: '33', logging.ERROR: '31', logging.CRITICAL: '31'}
    PAYLOADS = {'request', 'response', 'snapshot', 'config', 'clips', 'packet', 'decision', 'context', 'scheduled', 'monotonic_seconds'}

    def __init__(self, color=False, verbose=False):
        super().__init__()
        self.color = color
        self.verbose = verbose

    def format(self, record):
        timestamp = self.formatTime(record, '%H:%M:%S')
        level = f'{record.levelname:<7}'
        if self.color:
            level = f'\033[{self.COLORS.get(record.levelno, "0")}m{level}\033[0m'
        message = getattr(record, 'event', record.getMessage())
        fields = getattr(record, 'fields', {})
        details = []
        for key, value in fields.items():
            payload = key in self.PAYLOADS
            if payload and not self.verbose:
                continue
            rendered = json.dumps(value, ensure_ascii=False, default=str, separators=(',', ':'))
            if not payload and len(rendered) > 300:
                rendered = rendered[:297] + '...'
            details.append(f'{key}={rendered}')
        identifier = observation_id.get()
        if identifier is not None:
            details.append(f'obs={identifier[:8]}')
        line = f'{timestamp} {level} {message}'
        if details:
            line += ' ' + ' '.join(details)
        if record.exc_info:
            line += '\n' + self.formatException(record.exc_info)
        return line


def configure(path, verbose=False, debug=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8')
    file.setFormatter(JSONFormatter())
    console = logging.StreamHandler()
    console.setLevel(VERBOSE if verbose else logging.DEBUG if debug else logging.INFO)
    console.setFormatter(ConsoleFormatter(color=sys.stderr.isatty() and 'NO_COLOR' not in os.environ, verbose=verbose))
    log.setLevel(VERBOSE)
    log.propagate = False
    log.addHandler(file)
    log.addHandler(console)
    event('logging.started', path=str(path), console_level=logging.getLevelName(console.level))


def event(name, level=logging.INFO, *, exc_info=None, **fields):
    log.log(level, name, exc_info=exc_info,
            extra={'event': name, 'fields': {'monotonic_seconds': time.monotonic(), **fields}})


def trace(name, **fields):
    event(name, logging.DEBUG, **fields)


def verbose(name, **fields):
    event(name, VERBOSE, **fields)


@contextlib.contextmanager
def observation():
    token = observation_id.set(uuid.uuid4().hex)
    usage = {}
    usage_token = request_usage.set(usage)
    try:
        yield usage
    finally:
        request_usage.reset(usage_token)
        observation_id.reset(token)
