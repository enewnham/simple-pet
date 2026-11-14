import plistlib
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def agent(tmp_path: Path) -> dict[str, Any]:
    namespace = runpy.run_path(str(ROOT / "scripts/pet-mcp-agent.py"))
    install = namespace["install"]
    globals_ = install.__globals__
    globals_["ROOT"] = tmp_path / "pet checkout"
    checkout = globals_["ROOT"]
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/python").touch()
    (checkout / "pet.config.json").write_text("{}")
    return namespace


def test_install_serves_authenticated_http_without_shell_or_environment_secrets(
    tmp_path: Path, agent: dict[str, Any]
) -> None:
    plist = tmp_path / "home/Library/LaunchAgents/dev.simple-pet.mcp.plist"
    with (
        patch.object(Path, "home", return_value=tmp_path / "home"),
        patch("subprocess.run") as run,
    ):
        run.return_value = subprocess.CompletedProcess([], 1)
        agent["install"]("gui/501", "gui/501/dev.simple-pet.mcp", plist)
    payload = plistlib.loads(plist.read_bytes())
    checkout = tmp_path / "pet checkout"
    assert payload["Label"] == "dev.simple-pet.mcp"
    assert payload["ProgramArguments"] == [
        str(checkout / ".venv/bin/python"),
        "-m",
        "pet_daemon.mcp",
        "--http",
        "--host",
        "127.0.0.1",
        "--port",
        "18765",
    ]
    assert payload["WorkingDirectory"] == str(checkout)
    assert "EnvironmentVariables" not in payload
    assert payload["RunAtLoad"] is True
    assert payload["Umask"] == 0o077
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["StandardErrorPath"] == str(
        tmp_path / "home/Library/Logs/Simple Pet/mcp-stderr.log"
    )
    assert [call.args[0] for call in run.call_args_list] == [
        ["/bin/launchctl", "print", "gui/501/dev.simple-pet.mcp"],
        ["/bin/launchctl", "enable", "gui/501/dev.simple-pet.mcp"],
        ["/bin/launchctl", "bootstrap", "gui/501", str(plist)],
    ]


@pytest.mark.parametrize(
    ("action", "command"),
    [("restart", ["kickstart", "-k"]), ("status", ["print"])],
)
def test_controls_target_the_mcp_service(
    tmp_path: Path, agent: dict[str, Any], action: str, command: list[str]
) -> None:
    with (
        patch.object(Path, "home", return_value=tmp_path / "home"),
        patch("os.getuid", return_value=501),
        patch.object(sys, "argv", ["pet-mcp-agent.py", action]),
        patch("subprocess.run") as run,
    ):
        agent["main"]()
    run.assert_called_once_with(
        ["/bin/launchctl", *command, "gui/501/dev.simple-pet.mcp"], check=True
    )


def test_uninstall_stops_only_the_mcp_service_and_removes_its_plist(
    tmp_path: Path, agent: dict[str, Any]
) -> None:
    home = tmp_path / "home"
    plist = home / "Library/LaunchAgents/dev.simple-pet.mcp.plist"
    plist.parent.mkdir(parents=True)
    plist.touch()
    with (
        patch.object(Path, "home", return_value=home),
        patch("os.getuid", return_value=501),
        patch.object(sys, "argv", ["pet-mcp-agent.py", "uninstall"]),
        patch("subprocess.run") as run,
    ):
        run.return_value = subprocess.CompletedProcess([], 0)
        agent["main"]()
    assert not plist.exists()
    assert [call.args[0] for call in run.call_args_list] == [
        ["/bin/launchctl", "print", "gui/501/dev.simple-pet.mcp"],
        ["/bin/launchctl", "bootout", "gui/501/dev.simple-pet.mcp"],
    ]


def test_install_persists_the_selected_port(
    tmp_path: Path, agent: dict[str, Any]
) -> None:
    home = tmp_path / "home"
    with (
        patch.object(Path, "home", return_value=home),
        patch("os.getuid", return_value=501),
        patch.object(sys, "argv", ["pet-mcp-agent.py", "install", "--port", "8766"]),
        patch("subprocess.run") as run,
    ):
        run.return_value = subprocess.CompletedProcess([], 1)
        agent["main"]()
    plist = home / "Library/LaunchAgents/dev.simple-pet.mcp.plist"
    payload = plistlib.loads(plist.read_bytes())
    assert payload["ProgramArguments"][-2:] == ["--port", "8766"]


@pytest.mark.parametrize("port", ["0", "65536", "-1", "invalid"])
def test_invalid_ports_are_rejected_before_changing_the_service(
    agent: dict[str, Any], port: str
) -> None:
    with (
        patch.object(sys, "argv", ["pet-mcp-agent.py", "install", "--port", port]),
        patch("subprocess.run") as run,
        pytest.raises(SystemExit) as error,
    ):
        agent["main"]()
    assert error.value.code == 2
    run.assert_not_called()
