import argparse
import os
import plistlib
import subprocess
from pathlib import Path

LABEL = "dev.simple-pet"
ROOT = Path(__file__).resolve().parents[1]


def launchctl(*args: str) -> None:
    subprocess.run(["/bin/launchctl", *args], check=True)


def unload(service: str) -> None:
    result = subprocess.run(
        ["/bin/launchctl", "print", service], capture_output=True, check=False
    )
    if result.returncode == 0:
        launchctl("bootout", service)


def restart_mcp(domain: str) -> None:
    label = "dev.simple-pet.mcp"
    plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    if plist.is_file():
        launchctl("kickstart", "-k", f"{domain}/{label}")


def install(domain: str, service: str, plist: Path) -> None:
    for relative in (
        ".env",
        "pet.config.json",
        ".venv/bin/python",
        ".build/pet-overlay",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Follow the README quick start first.")

    logs = Path.home() / "Library/Logs/Simple Pet"
    logs.mkdir(parents=True, exist_ok=True)
    plist.parent.mkdir(parents=True, exist_ok=True)
    payload = plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": ["/bin/bash", str(ROOT / "scripts/pet-agent-run")],
            "WorkingDirectory": str(ROOT),
            "EnvironmentVariables": {"PATH": os.environ["PATH"]},
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "LimitLoadToSessionType": "Aqua",
            "ThrottleInterval": 10,
            "StandardOutPath": str(logs / "stdout.log"),
            "StandardErrorPath": str(logs / "stderr.log"),
        }
    )
    unload(service)
    plist.write_bytes(payload)
    launchctl("enable", service)
    launchctl("bootstrap", domain, str(plist))
    restart_mcp(domain)
    print(f"Installed {plist}")
    print(f"Startup logs: {logs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the Simple Pet LaunchAgent")
    parser.add_argument("action", choices=("install", "uninstall", "restart", "status"))
    args = parser.parse_args()
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"

    match args.action:
        case "install":
            install(domain, service, plist)
        case "uninstall":
            unload(service)
            plist.unlink(missing_ok=True)
            print(f"Uninstalled {plist}")
        case "restart":
            launchctl("kickstart", "-k", service)
            restart_mcp(domain)
        case "status":
            launchctl("print", service)


if __name__ == "__main__":
    main()
