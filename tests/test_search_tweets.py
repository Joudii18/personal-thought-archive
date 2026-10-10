"""Tests for ordinary keyword search using synthetic, non-private tweets."""

from __future__ import annotations

import unittest
from typing import Any

from src.search_tweets import get_post_type, search_tweets


def fake_tweet(
    tweet_id: str,
    text: str,
    created_at: str,
    *,
    likes: int = 0,
    language: str = "en",
    is_reply: bool = False,
    is_retweet: bool = False,
) -> dict[str, Any]:
    """Build a minimal cleaned tweet record for testing."""
    return {
        "id": tweet_id,
        "text": text,
        "created_at": created_at,
        "likes": likes,
        "retweets": 0,
        "language": language,
        "is_reply": is_reply,
        "is_retweet": is_retweet,
        "reply_to_tweet_id": "reply-target" if is_reply else None,
        "reply_to_user_id": "user-target" if is_reply else None,
        "source_file": "fake.js",
    }


class SearchTweetsTests(unittest.TestCase):
    """Verify search matching, filtering, classification, and sorting."""

    def setUp(self) -> None:
        self.tweets = [
            fake_tweet(
                "1",
                "Learning AI through small projects",
                "2024-01-02T10:00:00+00:00",
                likes=5,
            ),
            fake_tweet(
                "2",
                "مشروع صغير لتعلم البرمجة",
                "2025-03-04T10:00:00+00:00",
                likes=8,
                language="ar",
                is_reply=True,
            ),
            fake_tweet(
                "3",
                "RT @example: An AI research update",
                "2023-05-06T10:00:00+00:00",
                likes=1,
                is_retweet=True,
            ),
            fake_tweet(
                "4",
                "Another ai experiment",
                "2025-07-08T10:00:00+00:00",
                likes=12,
            ),
        ]

    def test_search_is_case_insensitive(self) -> None:
        matches = search_tweets(self.tweets, "AI")
        self.assertEqual([tweet["id"] for tweet in matches], ["4", "1", "3"])

    def test_search_supports_arabic(self) -> None:
        matches = search_tweets(self.tweets, "البرمجة")
        self.assertEqual([tweet["id"] for tweet in matches], ["2"])

    def test_filters_can_be_combined(self) -> None:
        matches = search_tweets(
            self.tweets,
            "مشروع",
            year=2025,
            language="AR",
            post_type="reply",
        )
        self.assertEqual([tweet["id"] for tweet in matches], ["2"])

    def test_most_liked_sort(self) -> None:
        matches = search_tweets(self.tweets, "ai", sort_by="most-liked")
        self.assertEqual([tweet["likes"] for tweet in matches], [12, 5, 1])

    def test_post_type_prefers_retweet_then_reply(self) -> None:
        self.assertEqual(get_post_type(self.tweets[0]), "original")
        self.assertEqual(get_post_type(self.tweets[1]), "reply")
        self.assertEqual(get_post_type(self.tweets[2]), "retweet")

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            search_tweets(self.tweets, "   ")


if __name__ == "__main__":
    unittest.main()
