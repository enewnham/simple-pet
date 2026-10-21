import ast
import unittest
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "pet_daemon"


def imports(path: Path) -> Iterator[tuple[str, str | None]]:
    package = ("pet_daemon", *path.relative_to(ROOT).parts[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for name in node.names:
                yield name.name, None
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else ()
            module = ".".join((*prefix, *(node.module or "").split("."))).rstrip(".")
            for name in node.names:
                yield module, name.name


class ArchitectureTests(unittest.TestCase):
    def assert_import_boundary(self, directory: str, forbidden: set[str]) -> None:
        for path in (ROOT / directory).rglob("*.py"):
            for module, name in imports(path):
                with self.subTest(
                    file=str(path.relative_to(ROOT)), module=module, name=name
                ):
                    self.assertFalse(
                        any(
                            module == value or module.startswith(value + ".")
                            for value in forbidden
                        ),
                        f"{path.relative_to(ROOT)} imports {module}",
                    )

    def test_work_rules_depend_on_domain_values_and_repository_ports(self) -> None:
        self.assert_import_boundary(
            "work",
            {
                "sqlite3",
                "subprocess",
                "socket",
                "asyncio",
                "anthropic",
                "httpx",
                "pet_daemon.storage",
                "pet_daemon.sources",
                "pet_daemon.commentary",
                "pet_daemon.attention",
                "pet_daemon.application",
                "pet_daemon.daemon",
                "pet_daemon.behavior",
                "pet_daemon.delivery",
            },
        )

    def test_source_adapters_emit_observations_without_storage_or_commentary_imports(
        self,
    ) -> None:
        self.assert_import_boundary(
            "sources",
            {
                "sqlite3",
                "pet_daemon.storage",
                "pet_daemon.commentary",
                "pet_daemon.attention",
                "pet_daemon.application",
                "pet_daemon.daemon",
            },
        )

    def test_commentary_consumes_selections_without_selecting_attention(self) -> None:
        forbidden = {
            "select",
            "choose_session",
            "select_subject",
            "choose_focus",
            "_session_tier",
            "_action_tier",
        }
        for path in (ROOT / "commentary").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for module, name in imports(path):
                if module == "pet_daemon.attention":
                    with self.subTest(file=path.name, imported=name):
                        self.assertEqual(name, "AttentionSelection")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    with self.subTest(file=path.name, function=node.name):
                        self.assertNotIn(node.name, forbidden)
                elif isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Attribute
                ):
                    with self.subTest(file=path.name, call=node.func.attr):
                        self.assertNotEqual(node.func.attr, "select")
