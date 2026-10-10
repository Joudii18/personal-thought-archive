"""Tests for Ask My Archive prompt construction and API output parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ask_archive import build_user_input, format_evidence  # noqa: E402
from openai_responses_client import extract_output_text  # noqa: E402


class AskArchiveTests(unittest.TestCase):
    def test_formats_numbered_evidence(self) -> None:
        tweets = [
            (
                {
                    "id": "1",
                    "text": "A project idea",
                    "created_at": "2024-01-01T12:00:00+00:00",
                    "is_reply": False,
                    "is_retweet": False,
                },
                0.75,
            )
        ]
        evidence = format_evidence(tweets, 1)
        self.assertIn("[Tweet 1]", evidence)
        self.assertIn("Date: 2024-01-01", evidence)
        self.assertIn("Text: A project idea", evidence)

    def test_separates_question_and_evidence(self) -> None:
        prompt = build_user_input("What did I build?", "[Tweet 1]\nEvidence")
        self.assertIn("QUESTION\nWhat did I build?", prompt)
        self.assertIn("TWEET EVIDENCE\n[Tweet 1]", prompt)

    def test_extracts_multiple_output_text_blocks(self) -> None:
        response = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "First"},
                        {"type": "output_text", "text": "Second"},
                    ]
                }
            ]
        }
        self.assertEqual(extract_output_text(response), "First\nSecond")

    def test_reports_incomplete_response_reason(self) -> None:
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            extract_output_text(response)

    def test_accepts_top_level_output_text(self) -> None:
        self.assertEqual(
            extract_output_text({"output_text": "  Answer  ", "output": []}),
            "Answer",
        )


if __name__ == "__main__":
    unittest.main()
