"""Tests for API batching and response validation without network calls."""

from __future__ import annotations

import unittest

import numpy as np

from src.openai_embeddings_client import embed_in_batches


class OpenAIEmbeddingClientTests(unittest.TestCase):
    """Verify batching with a synthetic request function."""

    def test_batches_preserve_order_and_accumulate_tokens(self) -> None:
        calls: list[list[str]] = []

        def fake_request(
            texts: list[str], model: str, dimensions: int, api_key: str
        ) -> tuple[np.ndarray, int]:
            calls.append(texts)
            rows = [[float(int(text)), 1.0] for text in texts]
            return np.asarray(rows, dtype=np.float32), len(texts) * 2

        vectors, tokens = embed_in_batches(
            ["1", "2", "3", "4", "5"],
            "fake-model",
            2,
            "fake-key",
            2,
            request_function=fake_request,
        )

        self.assertEqual(calls, [["1", "2"], ["3", "4"], ["5"]])
        self.assertEqual(vectors[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(tokens, 10)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no text"):
            embed_in_batches([], "fake-model", 2, "fake-key", 2)


if __name__ == "__main__":
    unittest.main()
