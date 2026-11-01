import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

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
