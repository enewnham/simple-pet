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
            "Checked parser. [omitted] Validated tests.",
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
            work_prose(report),
            "Built the explorer. [omitted] Checked the interactions.",
        )

    def test_marks_removed_examples_without_joining_their_introductions(self) -> None:
        self.assertEqual(
            prose_text(
                "Before:\n```python\nold = True\n```\nAfter:\n```python\nnew = True\n```\nBoth examples use the same interface."
            ),
            "Before:\n[omitted]\nAfter:\n[omitted]\nBoth examples use the same interface.",
        )

    def test_collapses_adjacent_removed_blocks(self) -> None:
        self.assertEqual(
            work_prose(
                "Checked the implementation.\n```\ncode\n```\n\n> Quoted pet speech\n\n| Heading |\n| --- |\n| Value |\n\n    more_code()\nValidated the result."
            ),
            "Checked the implementation. [omitted] Validated the result.",
        )

    def test_keeps_only_markers_between_retained_prose(self) -> None:
        self.assertEqual(
            work_prose("```\nfirst\n```\nChecked the implementation.\n```\nlast\n```"),
            "Checked the implementation.",
        )

    def test_does_not_mark_excluded_pet_speech_as_missing_evidence(self) -> None:
        self.assertEqual(
            prose_text(
                'Checked the inputs.\nThe pet said: "Quiet again."\nValidated the result.',
                excluded=("Quiet again.",),
            ),
            "Checked the inputs.\n\nValidated the result.",
        )

    def test_keeps_complete_sentences_and_marks_omitted_prose(self) -> None:
        self.assertEqual(
            work_prose("Built the explorer. Checked the interactions thoroughly.", 40),
            "Built the explorer. [omitted]",
        )

    def test_does_not_duplicate_a_marker_when_bounding_removed_examples(self) -> None:
        self.assertEqual(
            work_prose(
                "Before:\n```\nremoved\n```\nAfter the interface is checked.", 30
            ),
            "Before: [omitted]",
        )

    def test_does_not_treat_indented_blank_lines_as_removed_code(self) -> None:
        self.assertEqual(
            prose_text("Checked the inputs.\n    \nValidated the result."),
            "Checked the inputs.\n\nValidated the result.",
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
        self.assertEqual(work_prose("| Heading |\n| --- |\n| Value |"), "")
        self.assertEqual(work_prose("    code()"), "")
        self.assertEqual(
            work_prose('"Quiet again."\n```\nnoise\n```', excluded=("Quiet again.",)),
            "",
        )
