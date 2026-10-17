import io
import json
import logging
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pet_daemon.telemetry import ConsoleFormatter, JSONFormatter, configure, event, observation, request_usage, trace


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('pet-daemon')
        self.handlers = self.logger.handlers[:]
        self.level = self.logger.level
        self.propagate = self.logger.propagate
        self.logger.handlers = []

    def tearDown(self):
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers = self.handlers
        self.logger.setLevel(self.level)
        self.logger.propagate = self.propagate

    def test_console_shows_fields_and_correlation_without_full_payloads(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ConsoleFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        with observation():
            event('speech.delivered', text='A line\nwith a "quote"', session='codex:one', packet={'secret': 'full payload'})
        output = stream.getvalue()
        self.assertRegex(output, r'^\d{2}:\d{2}:\d{2} INFO')
        self.assertIn('speech.delivered', output)
        self.assertIn('session="codex:one"', output)
        self.assertIn('text="A line\\nwith a \\"quote\\""', output)
        self.assertRegex(output, r'obs=[a-f0-9]{8}')
        self.assertNotIn('full payload', output)
        self.assertNotIn('\033[', output)
        self.assertEqual(len(output.splitlines()), 1)

    def test_verbose_console_and_json_preserve_different_detail(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose), tempfile.TemporaryDirectory() as directory:
                stream = io.StringIO()
                path = Path(directory) / 'pet.jsonl'
                with patch('sys.stderr', stream):
                    configure(path, verbose)
                with observation():
                    trace('observation.request', reason='routine', request={'messages': ['full payload']})
                    event('observation.finished', outcome='silent', input_tokens=25)
                console = stream.getvalue()
                self.assertEqual('observation.request' in console, verbose)
                self.assertIn('outcome="silent"', console)
                self.assertNotIn('full payload', console)
                records = [json.loads(line) for line in path.read_text().splitlines()]
                request = next(record for record in records if record['event'] == 'observation.request')
                self.assertEqual(request['request'], {'messages': ['full payload']})
                finished = next(record for record in records if record['event'] == 'observation.finished')
                self.assertEqual(request['observation_id'], finished['observation_id'])
                for handler in self.logger.handlers:
                    handler.close()
                self.logger.handlers = []

    def test_color_requires_terminal_and_respects_no_color(self):
        for terminal, no_color, expected in ((True, False, True), (False, False, False), (True, True, False)):
            with self.subTest(terminal=terminal, no_color=no_color), tempfile.TemporaryDirectory() as directory:
                stream = io.StringIO()
                environment = {'NO_COLOR': ''} if no_color else {}
                with patch('sys.stderr', stream), patch.object(stream, 'isatty', return_value=terminal), patch.dict('os.environ', environment, clear=True):
                    configure(Path(directory) / 'pet.jsonl')
                    event('test.warning', level=logging.WARNING, reason='diagnostic')
                self.assertEqual('\033[' in stream.getvalue(), expected)
                for handler in self.logger.handlers:
                    handler.close()
                self.logger.handlers = []

    def test_observation_usage_is_isolated_and_restored(self):
        with observation() as outer:
            outer['routine'] = {'input_tokens': 25}
            with observation() as inner:
                self.assertEqual(inner, {})
                self.assertIs(request_usage.get(), inner)
            self.assertIs(request_usage.get(), outer)
        self.assertIsNone(request_usage.get())

    def test_json_includes_exception_and_full_fields(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        try:
            raise ValueError('broken fixture')
        except ValueError:
            event('observation.failed', level=logging.WARNING, exc_info=True, session='codex:one')
        record = json.loads(stream.getvalue())
        self.assertEqual(record['level'], 'WARNING')
        self.assertEqual(record['session'], 'codex:one')
        self.assertIn('ValueError: broken fixture', record['exception'])
