import json
import unittest

from pet_daemon.prompts import select_prompt


class PromptSelectionTests(unittest.TestCase):
    def test_short_prompt_keeps_its_exact_text(self) -> None:
        prompt = "  Fix 猫.py.\nKeep these lines.  "
        self.assertEqual(select_prompt(prompt), prompt)

    def test_late_request_survives_long_background(self) -> None:
        prompt = "Here is some background. " * 70 + "\nFix the parser for escaped JSON."
        selected = select_prompt(prompt)
        self.assertLessEqual(len(selected), 1000)
        self.assertIn("Fix the parser for escaped JSON.", selected)
        self.assertIn("Here is some background.", selected)
        self.assertIn("[omitted]", selected)

    def test_code_keywords_do_not_outrank_a_prose_question(self) -> None:
        prompt = (
            "Opening context.\n```python\n"
            + "fix = 1\n" * 160
            + "```\nWhy does the reader discard the final line?\nClosing context."
        )
        selected = select_prompt(prompt, limit=140)
        self.assertIn("Why does the reader discard the final line?", selected)
        self.assertNotIn("fix = 1", selected)

    def test_json_byte_budget_keeps_complete_unicode_characters(self) -> None:
        prompt = "猫" * 300 + "\nFix parser/猫.py for 🐈 input."
        selected = select_prompt(prompt, limit=1000, max_bytes=150)
        self.assertLessEqual(
            len(json.dumps(selected, ensure_ascii=False).encode("utf-8")), 150
        )
        self.assertIn("Fix parser/猫.py for 🐈 input.", selected)

    def test_large_first_anchor_leaves_room_for_final_request(self) -> None:
        prompt = "Background " + "x" * 340 + "\nFix the final parser bug."
        selected = select_prompt(prompt, limit=120)
        self.assertEqual(selected, "[omitted] Fix the final parser bug.")

    def test_fallback_marks_earlier_units_as_omitted(self) -> None:
        prompt = "Opening context.\n" + "Fix " + "x" * 500
        selected = select_prompt(prompt, limit=45)
        self.assertTrue(selected.startswith("[omitted] Fix "))
        self.assertTrue(selected.endswith("[omitted]"))
        self.assertLessEqual(len(selected), 45)

    def test_whitespace_only_prompt_obeys_json_byte_budget(self) -> None:
        selected = select_prompt(" " * 3000, limit=1000, max_bytes=40)
        self.assertLessEqual(len(selected), 1000)
        self.assertLessEqual(
            len(json.dumps(selected, ensure_ascii=False).encode("utf-8")), 40
        )
        self.assertIn("[omitted]", selected)
