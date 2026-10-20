import unittest

from pet_daemon.sentiment import NEGATIVE_THRESHOLD, sentiment_score
from pet_daemon.user_text import sentiment_excerpt


class SentimentTests(unittest.TestCase):
    def test_scores_criticism_without_diluting_it_with_the_explanation(self) -> None:
        text = (
            "See, this is why I think you've done a poor job here. "
            "Look at how property value schemas are set. "
            "The schema defines the maximum, and the values follow the schema."
        )
        self.assertLessEqual(
            sentiment_score(sentiment_excerpt(text)), NEGATIVE_THRESHOLD
        )
        self.assertLessEqual(
            sentiment_score("It freaking works, stop getting harped up on this."),
            NEGATIVE_THRESHOLD,
        )
        self.assertGreater(
            sentiment_score("Thank you, this looks great."), NEGATIVE_THRESHOLD
        )

    def test_excludes_code_quotes_and_embedded_agent_messages(self) -> None:
        text = (
            "Check the parser.\n> This is terrible and awful.\n"
            '```python\nerror = "This is terrible and awful."\n```\n'
            "~~~\nThis is terrible and awful.\n~~~\n"
            '<agent-message from="worker">This is terrible and awful.</agent-message>\n'
            "<task-notification>This is terrible and awful.</task-notification>\n"
            '<pasted_content id="one">This is terrible and awful.</pasted_content id="one">\n'
            "Use `fatal_error` as the name."
        )
        excerpt = sentiment_excerpt(text)
        self.assertEqual(excerpt, "Check the parser.\nUse  as the name.")
        self.assertEqual(sentiment_score(excerpt), 0)

    def test_bounds_unicode_excerpts_before_transport(self) -> None:
        text = "This is terrible.\n" + "🐈" * 2000 + "\nPlease fix the parser."
        excerpt = sentiment_excerpt(text)
        self.assertLessEqual(len(excerpt.encode("utf-8")), 600)
        self.assertIn("This is terrible.", excerpt)
        self.assertIn("Please fix the parser.", excerpt)


if __name__ == "__main__":
    unittest.main()
