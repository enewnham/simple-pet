import unittest

from pet_daemon.prose import prose_text, work_prose


class WorkProseTests(unittest.TestCase):
    def test_preserves_prose_lines_for_request_selection(self) -> None:
        self.assertEqual(
            prose_text("Background details\n> Quoted context\nFix the parser"),
            "Background details\nFix the parser",
        )

    def test_discards_tables_without_outer_pipes_and_indented_code(self) -> None:
        self.assertEqual(
            work_prose(
                "Checked parser.\nRequest | Response\n--- | ---\n"
                "Commentary | Invented worry\n\n"
                "    print('invented worry')\n\tprint('more noise')\nValidated tests."
            ),
            "Checked parser. Validated tests.",
        )

    def test_discards_quoted_code_and_tables_before_bounding(self) -> None:
        report = """Built the explorer.
> My mail is haunted.
```python
raise RuntimeError('ghost')
```
| Request | Response |
| --- | --- |
| Commentary | Silence |
Checked the interactions.
"""
        self.assertEqual(
            work_prose(report), "Built the explorer. Checked the interactions."
        )

    def test_keeps_complete_sentences_and_marks_omitted_prose(self) -> None:
        self.assertEqual(
            work_prose("Built the explorer. Checked the interactions thoroughly.", 40),
            "Built the explorer. [omitted]",
        )

    def test_excludes_known_pet_speech_from_plain_reports(self) -> None:
        self.assertEqual(
            work_prose(
                'The pet said: "Mail is haunted." Checked the inputs.',
                excluded=("Mail is haunted.",),
            ),
            "Checked the inputs.",
        )

    def test_excludes_a_multisentence_utterance(self) -> None:
        self.assertEqual(
            work_prose(
                'The pet said: "Mail is haunted. Quiet again." Checked the inputs.',
                excluded=("Mail is haunted. Quiet again.",),
            ),
            "Checked the inputs.",
        )

    def test_bounds_long_unpunctuated_reports(self) -> None:
        result = work_prose("word " * 100, 45)
        self.assertLessEqual(len(result), 45)
        self.assertTrue(result.endswith("[omitted]"))

    def test_omits_empty_inputs(self) -> None:
        self.assertEqual(work_prose(None), "")
        self.assertEqual(work_prose("```\nnoise\n```"), "")
