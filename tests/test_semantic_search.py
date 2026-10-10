"""Tests for semantic-search math and embedding-index safeguards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.embedding_utils import (
    dataset_fingerprint,
    load_embedding_index,
    save_embedding_index,
    verify_embedding_alignment,
)
from src.semantic_search import cosine_scores, semantic_rank


def fake_tweet(
    tweet_id: str,
    text: str,
    created_at: str,
    *,
    language: str = "en",
    is_reply: bool = False,
) -> dict[str, object]:
    """Return a synthetic cleaned record containing no private data."""
    return {
        "id": tweet_id,
        "text": text,
        "created_at": created_at,
        "likes": 0,
        "retweets": 0,
        "language": language,
        "is_reply": is_reply,
        "is_retweet": False,
        "reply_to_tweet_id": "target" if is_reply else None,
        "reply_to_user_id": "user" if is_reply else None,
        "source_file": "fake.js",
    }


class SemanticSearchTests(unittest.TestCase):
    """Verify vector ranking, filters, persistence, and stale-data checks."""

    def setUp(self) -> None:
        self.tweets = [
            fake_tweet("1", "Learning with projects", "2024-01-01T00:00:00+00:00"),
            fake_tweet(
                "2",
                "تعلم البرمجة",
                "2025-01-01T00:00:00+00:00",
                language="ar",
                is_reply=True,
            ),
            fake_tweet("3", "Cooking dinner", "2025-02-01T00:00:00+00:00"),
        ]
        self.embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
        )

    def test_cosine_scores_rank_directional_similarity(self) -> None:
        scores = cosine_scores(np.asarray([1.0, 0.0]), self.embeddings)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_semantic_rank_applies_filters(self) -> None:
        ranked = semantic_rank(
            self.tweets,
            self.embeddings,
            np.asarray([1.0, 0.0]),
            year=2025,
            language="AR",
            post_type="reply",
        )
        self.assertEqual([tweet["id"] for tweet, _ in ranked], ["2"])

    def test_index_round_trip(self) -> None:
        fingerprint = dataset_fingerprint(self.tweets)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.npz"
            save_embedding_index(
                path,
                [tweet["id"] for tweet in self.tweets],
                self.embeddings,
                "fake-model",
                fingerprint,
            )
            ids, matrix, model_name, saved_fingerprint = load_embedding_index(path)

        self.assertEqual(ids, ["1", "2", "3"])
        np.testing.assert_allclose(matrix, self.embeddings)
        self.assertEqual(model_name, "fake-model")
        self.assertEqual(saved_fingerprint, fingerprint)

    def test_changed_text_is_rejected(self) -> None:
        changed = [dict(tweet) for tweet in self.tweets]
        changed[0]["text"] = "Changed text"
        with self.assertRaisesRegex(ValueError, "text changed"):
            verify_embedding_alignment(
                changed,
                [tweet["id"] for tweet in changed],
                self.embeddings,
                dataset_fingerprint(self.tweets),
            )

    def test_reordered_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordering changed"):
            verify_embedding_alignment(
                self.tweets,
                ["2", "1", "3"],
                self.embeddings,
                dataset_fingerprint(self.tweets),
            )


if __name__ == "__main__":
    unittest.main()
