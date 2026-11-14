import argparse
import os
import plistlib
import re
import shutil
import signal
import subprocess
from pathlib import Path

from pet_daemon.tunnel import prepare_policy

LABEL = "dev.simple-pet.tunnel"
ROOT = Path(__file__).resolve().parents[1]


def launchctl(*args: str) -> None:
    subprocess.run(["/bin/launchctl", *args], check=True)


def unload(service: str) -> None:
    result = subprocess.run(
        ["/bin/launchctl", "print", service], capture_output=True, check=False
    )
    if result.returncode == 0:
        launchctl("bootout", service)


def install(domain: str, service: str, plist: Path, hostname: str) -> None:
    executable = shutil.which("ngrok")
    if executable is None:
        raise SystemExit(
            "Install ngrok and authenticate it before installing the tunnel."
        )
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        raise SystemExit("Run make setup before installing the tunnel.")
    prepare_policy(ROOT)
    logs = Path.home() / "Library/Logs/Simple Pet"
    logs.mkdir(parents=True, exist_ok=True)
    plist.parent.mkdir(parents=True, exist_ok=True)
    payload = plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [
                str(python),
                "-m",
                "scripts.pet-tunnel-agent",
                "run",
                "--domain",
                hostname,
                "--ngrok",
                str(Path(executable).absolute()),
            ],
            "WorkingDirectory": str(ROOT),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "LimitLoadToSessionType": "Aqua",
            "ThrottleInterval": 10,
            "Umask": 0o077,
            "StandardOutPath": str(logs / "tunnel-stdout.log"),
            "StandardErrorPath": str(logs / "tunnel-stderr.log"),
        }
    )
    unload(service)
    plist.write_bytes(payload)
    launchctl("enable", service)
    launchctl("bootstrap", domain, str(plist))
    print(f"Installed {plist}")
    print(f"Startup logs: {logs}")


def run_tunnel(executable: str, hostname: str) -> int:
    try:
        policy = prepare_policy(ROOT)
        process = subprocess.Popen(
            [
                executable,
                "http",
                "http://127.0.0.1:18765",
                "--url",
                f"https://{hostname}",
                "--inspect=false",
                "--log=false",
                "--traffic-policy-file",
                str(policy),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError, ValueError:
        print("Tunnel startup failed.", flush=True)
        return 1

    def stop(signum: int, _frame: object) -> None:
        process.send_signal(signum)

    handlers = {
        kind: signal.signal(kind, stop) for kind in (signal.SIGTERM, signal.SIGINT)
    }
    print("Tunnel process started.", flush=True)
    try:
        code = process.wait()
    finally:
        for kind, handler in handlers.items():
            signal.signal(kind, handler)
    print(f"Tunnel process exited with status {code}.", flush=True)
    return code if code >= 0 else 128 - code


def hostname(value: str) -> str:
    label = r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    if len(value) > 253 or re.fullmatch(rf"{label}(?:\.{label})+", value) is None:
        raise argparse.ArgumentTypeError(
            "Domain must be a DNS hostname without a URL or port"
        )
    return value.lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the Simple Pet tunnel LaunchAgent"
    )
    parser.add_argument(
        "action", choices=("install", "uninstall", "restart", "status", "run")
    )
    parser.add_argument("--domain", type=hostname)
    parser.add_argument("--ngrok", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action in ("install", "run") and args.domain is None:
        parser.error("--domain is required for this action")
    if args.action == "run":
        if args.ngrok is None:
            parser.error("The internal run action requires --ngrok")
        raise SystemExit(run_tunnel(args.ngrok, args.domain))
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    match args.action:
        case "install":
            install(domain, service, plist, args.domain)
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
