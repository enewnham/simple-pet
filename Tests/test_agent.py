import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


class AgentTests(unittest.TestCase):
    def test_launcher_exports_environment_on_each_start_from_another_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="pet-agent-") as directory:
            checkout = Path(directory) / "pet checkout"
            scripts = checkout / "scripts"
            scripts.mkdir(parents=True)
            launcher = scripts / "pet-agent-run"
            shutil.copyfile(ROOT / "scripts/pet-agent-run", launcher)
            python = checkout / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                '#!/bin/bash\nprintf "%s\\n" "$PWD" "$PET_AGENT_TEST" "$@"\n'
            )
            python.chmod(0o755)

            for value in ("first value", "changed value"):
                (checkout / ".env").write_text(f"PET_AGENT_TEST='{value}'\n")
                result = subprocess.run(
                    ["/bin/bash", str(launcher)],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                lines = result.stdout.splitlines()
                self.assertEqual(Path(lines[0]).resolve(), checkout.resolve())
                self.assertEqual(lines[1:], [value, "-m", "pet_daemon"])


@pytest.fixture
def agent(tmp_path: Path) -> dict[str, Any]:
    namespace = runpy.run_path(str(ROOT / "scripts/pet-agent.py"))
    globals_ = namespace["install"].__globals__
    checkout = tmp_path / "pet checkout"
    globals_["ROOT"] = checkout
    for relative in (
        ".env",
        "pet.config.json",
        ".venv/bin/python",
        ".build/pet-overlay",
    ):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return namespace


@pytest.mark.parametrize("action", ["install", "restart"])
@pytest.mark.parametrize("mcp_installed", [True, False])
def test_daemon_start_restarts_installed_mcp(
    tmp_path: Path, agent: dict[str, Any], action: str, mcp_installed: bool
) -> None:
    home = tmp_path / "home"
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    if mcp_installed:
        (agents / "dev.simple-pet.mcp.plist").touch()
    with (
        patch.object(Path, "home", return_value=home),
        patch("os.getuid", return_value=501),
        patch.object(sys, "argv", ["pet-agent.py", action]),
        patch("subprocess.run") as run,
    ):
        run.return_value = subprocess.CompletedProcess([], 0)
        agent["main"]()
    daemon = "gui/501/dev.simple-pet"
    expected = (
        [
            ["/bin/launchctl", "print", daemon],
            ["/bin/launchctl", "bootout", daemon],
            ["/bin/launchctl", "enable", daemon],
            [
                "/bin/launchctl",
                "bootstrap",
                "gui/501",
                str(agents / "dev.simple-pet.plist"),
            ],
        ]
        if action == "install"
        else [["/bin/launchctl", "kickstart", "-k", daemon]]
    )
    if mcp_installed:
        expected.append(
            ["/bin/launchctl", "kickstart", "-k", "gui/501/dev.simple-pet.mcp"]
        )
    assert [call.args[0] for call in run.call_args_list] == expected
    assert all(
        call.kwargs["check"]
        for call in run.call_args_list
        if call.args[0][1] != "print"
    )
