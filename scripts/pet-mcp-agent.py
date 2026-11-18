import argparse
import os
import plistlib
import subprocess
from pathlib import Path

LABEL = "dev.simple-pet.mcp"
ROOT = Path(__file__).resolve().parents[1]


def launchctl(*args: str) -> None:
    subprocess.run(["/bin/launchctl", *args], check=True)


def unload(service: str) -> None:
    result = subprocess.run(
        ["/bin/launchctl", "print", service], capture_output=True, check=False
    )
    if result.returncode == 0:
        launchctl("bootout", service)


def install(
    domain: str,
    service: str,
    plist: Path,
    port: int = 18765,
    write_token_file: Path | None = None,
) -> None:
    for relative in ("pet.config.json", ".venv/bin/python"):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Run make setup and configure the pet first."
            )
    logs = Path.home() / "Library/Logs/Simple Pet"
    logs.mkdir(parents=True, exist_ok=True)
    plist.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "pet_daemon.mcp",
        "--http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if write_token_file is not None:
        arguments.extend(
            ("--write-token-file", str(write_token_file.expanduser().absolute()))
        )
    payload = plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": arguments,
            "WorkingDirectory": str(ROOT),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "LimitLoadToSessionType": "Aqua",
            "ThrottleInterval": 10,
            "Umask": 0o077,
            "StandardOutPath": str(logs / "mcp-stdout.log"),
            "StandardErrorPath": str(logs / "mcp-stderr.log"),
        }
    )
    unload(service)
    plist.write_bytes(payload)
    launchctl("enable", service)
    launchctl("bootstrap", domain, str(plist))
    print(f"Installed {plist}")
    print(f"Startup logs: {logs}")


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the Simple Pet MCP LaunchAgent"
    )
    parser.add_argument("action", choices=("install", "uninstall", "restart", "status"))
    parser.add_argument("--port", type=port_number, default=18765)
    parser.add_argument("--write-token-file", type=Path)
    args = parser.parse_args()
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"

    match args.action:
        case "install":
            install(domain, service, plist, args.port, args.write_token_file)
        case "uninstall":
            unload(service)
            plist.unlink(missing_ok=True)
            print(f"Uninstalled {plist}")
        case "restart":
            launchctl("kickstart", "-k", service)
        case "status":
            launchctl("print", service)


if __name__ == "__main__":
    main()
