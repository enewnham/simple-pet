import asyncio
import math
import threading
import unittest
from unittest.mock import patch

import objc
from NaturalLanguage import NLEmbedding, NLLanguageEnglish  # ty: ignore[unresolved-import]

from pet_daemon.commentary.repetition import repeat_match, similarity


class RepetitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_repeats_ignore_case_spacing_and_punctuation(self) -> None:
        result = await repeat_match(
            "  THE parser checks passed! ",
            ["Other remark", "The parser checks passed."],
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
                result = await repeat_match(text, [previous], threshold=-1)
                self.assertIsNone(result)

    async def test_unavailable_embedding_preserves_commentary(self) -> None:
        with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
            embedding.sentenceEmbeddingForLanguage_.return_value = None
            result = await repeat_match(
                "The build failed again.", ["The build failed."]
            )
        self.assertIsNone(result)

    async def test_missing_vectors_preserve_commentary(self) -> None:
        for vectors in ((None,), ([1.0, 0.0], None)):
            with self.subTest(vectors=vectors):
                with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
                    model = embedding.sentenceEmbeddingForLanguage_.return_value
                    model.vectorForString_.side_effect = vectors
                    result = await repeat_match(
                        "The build failed again.", ["The build failed."]
                    )
                self.assertIsNone(result)

    async def test_framework_errors_preserve_commentary(self) -> None:
        for operation in ("model", "vector"):
            with self.subTest(operation=operation):
                with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
                    lookup = embedding.sentenceEmbeddingForLanguage_
                    target = (
                        lookup
                        if operation == "model"
                        else lookup.return_value.vectorForString_
                    )
                    target.side_effect = objc.error("Embedding lookup failed")  # ty: ignore[unresolved-attribute]
                    result = await repeat_match(
                        "The build failed again.", ["The build failed."]
                    )
                self.assertIsNone(result)

    async def test_semantic_matches_select_the_highest_score_and_history_index(
        self,
    ) -> None:
        with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
            model = embedding.sentenceEmbeddingForLanguage_.return_value
            model.vectorForString_.side_effect = (
                [1.0, 0.0],
                [0.95, math.sqrt(1 - 0.95**2)],
                [1.0, 0.0],
            )
            result = await repeat_match(
                "The build failed again.",
                ["The tests passed.", "The build failed.", "The build still failed."],
            )
        assert result is not None
        self.assertEqual(
            (result.index, result.similarity, result.method), (2, 1.0, "semantic")
        )

    async def test_scores_below_the_requested_threshold_preserve_commentary(
        self,
    ) -> None:
        with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
            model = embedding.sentenceEmbeddingForLanguage_.return_value
            model.vectorForString_.side_effect = (
                [1.0, 0.0],
                [0.95, math.sqrt(1 - 0.95**2)],
            )
            result = await repeat_match(
                "The build failed again.", ["The build failed."], threshold=0.99
            )
        self.assertIsNone(result)

    async def test_embedding_lookup_keeps_the_event_loop_responsive(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def vector(text: str) -> list[float]:
            entered.set()
            if not release.wait(2):
                raise TimeoutError("The test did not release the embedding lookup")
            return [1.0, 0.0]

        with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
            model = embedding.sentenceEmbeddingForLanguage_.return_value
            model.vectorForString_.side_effect = vector
            task = asyncio.create_task(
                repeat_match("The build failed again.", ["The build failed."])
            )
            try:
                await asyncio.wait_for(self.wait_for_worker(entered), 1)
                marker = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(marker.set_result, None)
                await asyncio.wait_for(marker, 1)
                self.assertFalse(task.done())
            finally:
                release.set()
                result = await asyncio.wait_for(task, 1)
        self.assertIsNotNone(result)

    async def test_caller_cancellation_propagates_during_embedding_lookup(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def vector(text: str) -> list[float]:
            entered.set()
            try:
                if not release.wait(2):
                    raise TimeoutError("The test did not release the embedding lookup")
                return [1.0, 0.0]
            finally:
                finished.set()

        with patch("pet_daemon.commentary.repetition.NLEmbedding") as embedding:
            model = embedding.sentenceEmbeddingForLanguage_.return_value
            model.vectorForString_.side_effect = vector
            task = asyncio.create_task(
                repeat_match("The build failed again.", ["The build failed."])
            )
            try:
                await asyncio.wait_for(self.wait_for_worker(entered), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
            finally:
                release.set()
                await asyncio.wait_for(self.wait_for_worker(finished), 1)
                if not task.done():
                    await asyncio.wait_for(task, 1)

    @staticmethod
    async def wait_for_worker(signal: threading.Event) -> None:
        while not signal.is_set():
            await asyncio.sleep(0.001)


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity_accounts_for_direction_and_magnitude(self) -> None:
        for first, second, expected in (
            ([1.0, 0.0], [3.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
            ([1.0, 0.0], [-1.0, 0.0], -1.0),
            ([1.0, 1.0], [1.0, 0.0], 1 / math.sqrt(2)),
        ):
            with self.subTest(first=first, second=second):
                value = similarity(first, second)
                assert value is not None
                self.assertAlmostEqual(value, expected)

    def test_invalid_vectors_have_no_similarity(self) -> None:
        for first, second in (
            ([], []),
            ([1.0], [1.0, 2.0]),
            ([0.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.0], [0.0, 0.0]),
            ([math.nan], [1.0]),
            ([math.inf], [1.0]),
            ([1.0], [math.inf]),
        ):
            with self.subTest(first=first, second=second):
                self.assertIsNone(similarity(first, second))


class NativeRepetitionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with objc.autorelease_pool():
            embedding = NLEmbedding.sentenceEmbeddingForLanguage_(NLLanguageEnglish)
            if embedding is None:
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
        text = "There are 16 unstaged edits."
        previous = "There are 15 unstaged edits."
        with objc.autorelease_pool():
            embedding = NLEmbedding.sentenceEmbeddingForLanguage_(NLLanguageEnglish)
            assert embedding is not None
            score = similarity(
                embedding.vectorForString_(text), embedding.vectorForString_(previous)
            )
        assert score is not None
        self.assertGreater(score, 0.9)
        self.assertIsNone(await repeat_match(text, [previous]))
