import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from pet_daemon.commentary.repetition import SIMILARITY_BINARY, repeat_match


class RepetitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_repeats_ignore_case_spacing_and_punctuation(self) -> None:
        result = await repeat_match(
            "  THE parser checks passed! ",
            ["Other remark", "The parser checks passed."],
            executable=Path("/missing-similarity-helper"),
        )
        assert result is not None
        self.assertEqual(
            (result.index, result.similarity, result.method), (1, 1, "exact")
        )

    async def test_changed_numbers_outcomes_and_negation_preserve_new_evidence(
        self,
    ) -> None:
        for text, previous in (
            ("There are 16 unstaged edits.", "There are 15 unstaged edits."),
            ("The tests passed.", "The tests failed."),
            ("The pull request has merged.", "The pull request has not merged."),
            ("The tests haven't passed.", "The tests passed."),
            ("The value is -1.", "The value is 1."),
            ("The version is 1.2.", "The version is 12."),
            ("There are 15 staged edits.", "There are 15 unstaged edits."),
        ):
            with self.subTest(text=text):
                result = await repeat_match(
                    text,
                    [previous],
                    executable=Path("/missing-similarity-helper"),
                    threshold=-1,
                )
                self.assertIsNone(result)

    async def test_absent_helper_preserves_commentary(self) -> None:
        result = await repeat_match(
            "The build failed again.",
            ["The build failed."],
            executable=Path("/missing-similarity-helper"),
        )
        self.assertIsNone(result)

    async def test_unavailable_embedding_preserves_commentary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "unavailable"
            helper.write_text(
                '#!/bin/sh\nprintf \'{"available":false,"similarities":[null]}\'\n'
            )
            helper.chmod(0o755)
            result = await repeat_match(
                "The build failed again.", ["The build failed."], executable=helper
            )
            self.assertIsNone(result)

    async def test_timeout_stops_the_helper_and_preserves_commentary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "slow"
            helper.write_text("#!/bin/sh\nexec sleep 30\n")
            helper.chmod(0o755)
            result = await asyncio.wait_for(
                repeat_match(
                    "The build failed again.",
                    ["The build failed."],
                    executable=helper,
                    timeout=0.02,
                ),
                1,
            )
            self.assertIsNone(result)

    async def test_cancellation_reaps_the_native_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "slow"
            pid_file = Path(directory) / "pid"
            helper.write_text(
                f"#!/bin/sh\nprintf '%s' \"$$\" > '{pid_file}'\nexec sleep 30\n"
            )
            helper.chmod(0o755)
            task = asyncio.create_task(
                repeat_match(
                    "The build failed again.", ["The build failed."], executable=helper
                )
            )

            async def entered() -> None:
                while not pid_file.exists():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(entered(), 1)
            pid = int(pid_file.read_text())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


class NativeRepetitionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            [str(SIMILARITY_BINARY)],
            input=json.dumps({"text": "The build failed.", "previous": []}),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        if not json.loads(result.stdout)["available"]:
            raise unittest.SkipTest(
                "The local English sentence embedding is unavailable"
            )

    async def test_close_paraphrases_match_through_the_native_boundary(self) -> None:
        result = await repeat_match(
            "The parser build failed again.",
            ["A quiet break.", "The parser build failed."],
        )
        assert result is not None
        self.assertEqual((result.index, result.method), (1, "semantic"))
        self.assertGreaterEqual(result.similarity, 0.9)

    async def test_different_work_stays_distinguishable(self) -> None:
        result = await repeat_match(
            "The parser failed to compile.", ["The tests are still running."]
        )
        self.assertIsNone(result)

    async def test_same_result_for_a_different_subject_remains_new_evidence(
        self,
    ) -> None:
        result = await repeat_match(
            "The parser tests passed.", ["The backend tests passed."]
        )
        self.assertIsNone(result)

    async def test_added_subject_qualifier_preserves_new_evidence(self) -> None:
        result = await repeat_match(
            "The backend parser tests passed.", ["The parser tests passed."]
        )
        self.assertIsNone(result)

    async def test_native_high_similarity_does_not_suppress_changed_counts(
        self,
    ) -> None:
        data = {
            "text": "There are 16 unstaged edits.",
            "previous": ["There are 15 unstaged edits."],
        }
        result = await asyncio.create_subprocess_exec(
            str(SIMILARITY_BINARY),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        output, _ = await result.communicate(json.dumps(data).encode())
        self.assertGreater(json.loads(output)["similarities"][0], 0.9)
        self.assertIsNone(await repeat_match(data["text"], data["previous"]))
