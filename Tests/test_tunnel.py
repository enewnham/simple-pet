import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from pet_daemon.tunnel import (
    SECRET_PLACEHOLDER,
    Route,
    load_routes,
    prepare_policy,
    render_policy,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "examples/pet-mcp-ngrok.yml"


class TunnelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        examples = self.root / "examples"
        examples.mkdir()
        (examples / TEMPLATE.name).write_text(TEMPLATE.read_text())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_policy_retains_a_private_stable_secret_without_printing(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            path = prepare_policy(self.root)
            first = path.read_text()
            prepare_policy(self.root)
        secret_path = self.root / "state/pet-mcp-tunnel.secret"
        secret = secret_path.read_text().strip()
        self.assertGreaterEqual(len(secret), 32)
        self.assertEqual(path, self.root / "state/pet-mcp-ngrok.yml")
        self.assertEqual(path.read_text(), first)
        self.assertIn(f"!= [{json.dumps(secret)}]", first)
        self.assertNotIn(SECRET_PLACEHOLDER, first)
        self.assertEqual(output.getvalue(), "")
        for artifact in (path, secret_path):
            metadata = artifact.stat()
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_uid, os.getuid())
        self.assertEqual(set(path.parent.iterdir()), {path, secret_path})

    def test_policy_encodes_secret_as_one_expression_string(self) -> None:
        secret = '"] || true || ["injected\\nsecond line'
        template = TEMPLATE.read_text()
        policy = render_policy(template, secret)
        prefix, suffix = render_policy(template, SECRET_PLACEHOLDER).split(
            json.dumps(SECRET_PLACEHOLDER)
        )
        literal = policy[len(prefix) : -len(suffix)]
        self.assertEqual(json.loads(literal), secret)
        self.assertNotIn("\nsecond line", policy)
        self.assertIn('\\"', literal)

    def test_policy_checks_the_complete_header_value_list_before_routing(self) -> None:
        template = TEMPLATE.read_text()
        self.assertIn("!('x-pet-tunnel-key' in req.headers)", template)
        self.assertIn(
            "req.headers['x-pet-tunnel-key'] != [" + SECRET_PLACEHOLDER + "]",
            template,
        )
        self.assertLess(
            template.index(SECRET_PLACEHOLDER), template.index("req.url.path")
        )
        self.assertIn("req.url.path != '/mcp'", template)
        self.assertIn("req.method in ['POST', 'GET', 'DELETE']", template)
        self.assertIn('host: "127.0.0.1:18765"', template)

    def test_routes_forward_exact_paths_after_the_header_check(self) -> None:
        (self.root / "state").mkdir(mode=0o700)
        (self.root / "state/pet-tunnel-routes.json").write_text(
            json.dumps(
                [{"path": "/agent-rc/mcp", "forward": "https://agent-rc.internal"}]
            )
        )
        policy = prepare_policy(self.root).read_text()
        rule = (
            "req.url.path == '/agent-rc/mcp' && req.method in ['POST', 'GET', 'DELETE']"
        )
        self.assertIn(rule, policy)
        self.assertIn('"url": "https://agent-rc.internal"', policy)
        self.assertLess(policy.index("x-pet-tunnel-key"), policy.index(rule))
        self.assertLess(policy.index(rule), policy.index("req.url.path != '/mcp'"))

    def test_without_routes_the_policy_serves_only_mcp(self) -> None:
        policy = render_policy(TEMPLATE.read_text(), "s" * 40)
        self.assertNotIn("forward-internal", policy)
        self.assertNotIn("__PET_TUNNEL_ROUTES__", policy)

    def test_routes_reject_unsafe_paths_and_public_targets(self) -> None:
        for path, forward in (
            ("/mcp", "https://agent-rc.internal"),
            ("/a' || true || '", "https://agent-rc.internal"),
            ("relative", "https://agent-rc.internal"),
            ("/agent-rc/mcp", "https://example.com"),
            ("/agent-rc/mcp", "http://agent-rc.internal"),
        ):
            with (
                self.subTest(path=path, forward=forward),
                self.assertRaises(ValueError),
            ):
                Route(path, forward)
        routes = self.root / "routes.json"
        routes.write_text(
            json.dumps([{"path": "/a", "forward": "https://a.internal"}] * 2)
        )
        with self.assertRaises(ValueError):
            load_routes(routes)

    def test_existing_secret_requires_private_permissions_and_regular_file(
        self,
    ) -> None:
        path = prepare_policy(self.root)
        secret_path = self.root / "state/pet-mcp-tunnel.secret"
        secret = secret_path.read_text().strip()
        secret_path.chmod(0o644)
        with self.assertRaises(ValueError) as rejected:
            prepare_policy(self.root)
        self.assertNotIn(secret, str(rejected.exception))
        secret_path.chmod(0o600)
        source = self.root / "original-secret"
        secret_path.rename(source)
        secret_path.symlink_to(source)
        with self.assertRaises(OSError):
            prepare_policy(self.root)
        self.assertTrue(path.is_file())

    def test_invalid_template_does_not_include_secret_in_error(self) -> None:
        secret = "private-value"
        routes = "  # __PET_TUNNEL_ROUTES__\n"
        for template in (
            "no placeholder",
            SECRET_PLACEHOLDER * 2 + routes,
            SECRET_PLACEHOLDER,
        ):
            with self.assertRaises(ValueError) as rejected:
                render_policy(template, secret)
            self.assertNotIn(secret, str(rejected.exception))
