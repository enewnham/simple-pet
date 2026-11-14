import plistlib
import runpy
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "pet-example.ngrok-free.dev"


@pytest.fixture
def agent(tmp_path: Path) -> dict[str, Any]:
    namespace = runpy.run_path(str(ROOT / "scripts/pet-tunnel-agent.py"))
    globals_ = namespace["install"].__globals__
    checkout = tmp_path / "pet checkout"
    globals_["ROOT"] = checkout
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/python").touch()
    (checkout / "examples").mkdir()
    (checkout / "examples/pet-mcp-ngrok.yml").write_bytes(
        (ROOT / "examples/pet-mcp-ngrok.yml").read_bytes()
    )
    return namespace


def test_install_persists_domain_and_binary_with_private_logs(
    tmp_path: Path, agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    with (
        patch.object(Path, "home", return_value=home),
        patch("os.getuid", return_value=501),
        patch.object(
            sys, "argv", ["pet-tunnel-agent.py", "install", "--domain", DOMAIN]
        ),
        patch("shutil.which", return_value="/opt/homebrew/bin/ngrok"),
        patch("subprocess.run") as run,
    ):
        run.return_value = subprocess.CompletedProcess([], 1)
        agent["main"]()
    plist = home / "Library/LaunchAgents/dev.simple-pet.tunnel.plist"
    payload = plistlib.loads(plist.read_bytes())
    checkout = tmp_path / "pet checkout"
    assert payload["ProgramArguments"] == [
        str(checkout / ".venv/bin/python"),
        "-m",
        "scripts.pet-tunnel-agent",
        "run",
        "--domain",
        DOMAIN,
        "--ngrok",
        "/opt/homebrew/bin/ngrok",
    ]
    assert payload["WorkingDirectory"] == str(checkout)
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["RunAtLoad"] is True
    assert payload["Umask"] == 0o077
    assert payload["StandardErrorPath"] == str(
        home / "Library/Logs/Simple Pet/tunnel-stderr.log"
    )
    secret = (checkout / "state/pet-mcp-tunnel.secret").read_text().strip()
    assert secret not in plist.read_text()
    assert secret not in capsys.readouterr().out
    assert [call.args[0] for call in run.call_args_list] == [
        ["/bin/launchctl", "print", "gui/501/dev.simple-pet.tunnel"],
        ["/bin/launchctl", "enable", "gui/501/dev.simple-pet.tunnel"],
        ["/bin/launchctl", "bootstrap", "gui/501", str(plist)],
    ]


@pytest.mark.parametrize(
    "domain",
    [
        "https://example.com",
        "example.com:80",
        "example.com/path",
        "-bad.example",
        "a..example",
        "a" * 64 + ".example",
    ],
)
def test_invalid_domains_are_rejected_before_service_changes(
    agent: dict[str, Any], domain: str
) -> None:
    with (
        patch.object(
            sys, "argv", ["pet-tunnel-agent.py", "install", "--domain", domain]
        ),
        patch("subprocess.run") as run,
        pytest.raises(SystemExit) as error,
    ):
        agent["main"]()
    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize(
    ("action", "command"), [("restart", ["kickstart", "-k"]), ("status", ["print"])]
)
def test_controls_target_the_tunnel_service(
    agent: dict[str, Any], action: str, command: list[str]
) -> None:
    with (
        patch("os.getuid", return_value=501),
        patch.object(sys, "argv", ["pet-tunnel-agent.py", action]),
        patch("subprocess.run") as run,
    ):
        agent["main"]()
    run.assert_called_once_with(
        ["/bin/launchctl", *command, "gui/501/dev.simple-pet.tunnel"], check=True
    )


def test_run_discards_ngrok_output_and_forwards_termination(
    tmp_path: Path, agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("subprocess.Popen") as popen:
        process = popen.return_value

        def wait() -> int:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
            return -signal.SIGTERM

        process.wait.side_effect = wait
        code = agent["run_tunnel"]("/opt/homebrew/bin/ngrok", DOMAIN)
    args = popen.call_args.args[0]
    assert args == [
        "/opt/homebrew/bin/ngrok",
        "http",
        "http://127.0.0.1:18765",
        "--url",
        f"https://{DOMAIN}",
        "--inspect=false",
        "--log=false",
        "--traffic-policy-file",
        str(tmp_path / "pet checkout/state/pet-mcp-ngrok.yml"),
    ]
    assert popen.call_args.kwargs == {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    process.send_signal.assert_called_once_with(signal.SIGTERM)
    assert code == 143
    secret = (tmp_path / "pet checkout/state/pet-mcp-tunnel.secret").read_text().strip()
    assert secret not in capsys.readouterr().out


def test_startup_errors_never_print_policy_contents(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("subprocess.Popen", side_effect=OSError("secret policy contents")):
        assert agent["run_tunnel"]("/missing/ngrok", DOMAIN) == 1
    captured = capsys.readouterr()
    assert captured.out == "Tunnel startup failed.\n"
    assert captured.err == ""


def test_uninstall_removes_only_the_tunnel_service(
    tmp_path: Path, agent: dict[str, Any]
) -> None:
    home = tmp_path / "home"
    plist = home / "Library/LaunchAgents/dev.simple-pet.tunnel.plist"
    plist.parent.mkdir(parents=True)
    plist.touch()
    with (
        patch.object(Path, "home", return_value=home),
        patch("os.getuid", return_value=501),
        patch.object(sys, "argv", ["pet-tunnel-agent.py", "uninstall"]),
        patch("subprocess.run") as run,
    ):
        run.return_value = subprocess.CompletedProcess([], 0)
        agent["main"]()
    assert not plist.exists()
    assert [call.args[0] for call in run.call_args_list] == [
        ["/bin/launchctl", "print", "gui/501/dev.simple-pet.tunnel"],
        ["/bin/launchctl", "bootout", "gui/501/dev.simple-pet.tunnel"],
    ]
