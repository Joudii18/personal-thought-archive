"""Tests for writing-assistant prompt construction; no network calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from writing_assistant import build_user_input, format_examples  # noqa: E402


class WritingAssistantTests(unittest.TestCase):
    def test_formats_numbered_examples(self) -> None:
        ranked = [
            (
                {
                    "text": "Learning by building",
                    "created_at": "2024-01-01T12:00:00+00:00",
                },
                0.7,
            )
        ]
        examples = format_examples(ranked)
        self.assertIn("[Example 1]", examples)
        self.assertIn("Date: 2024-01-01", examples)
        self.assertIn("Text: Learning by building", examples)

    def test_separates_thought_from_examples(self) -> None:
        user_input = build_user_input("My own thought", "[Example 1]\nOld post")
        self.assertIn("ROUGH THOUGHT\nMy own thought", user_input)
        self.assertIn("STYLE EXAMPLES\n[Example 1]", user_input)


if __name__ == "__main__":
    unittest.main()
