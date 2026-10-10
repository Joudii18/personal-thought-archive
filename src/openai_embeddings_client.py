"""Minimal OpenAI Embeddings API client using Python's standard library."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np


EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_API_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 384


def get_api_key() -> str:
    """Read the API key from the environment without printing or storing it."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in your terminal before "
            "running the API embedding commands."
        )
    return api_key


def request_embeddings(
    texts: list[str],
    model: str,
    dimensions: int,
    api_key: str,
    *,
    timeout: float = 120,
    max_attempts: int = 4,
) -> tuple[np.ndarray, int]:
    """Request one embedding per text and return vectors plus token usage."""
    if not texts or not all(isinstance(text, str) and text for text in texts):
        raise ValueError("Embedding input must contain non-empty strings.")
    if dimensions < 1:
        raise ValueError("Embedding dimensions must be positive.")

    payload = json.dumps(
        {
            "input": texts,
            "model": model,
            "dimensions": dimensions,
            "encoding_format": "float",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        EMBEDDINGS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"OpenAI Embeddings API returned HTTP {error.code}."
                ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error

        if attempt + 1 == max_attempts:
            raise RuntimeError(
                "OpenAI Embeddings API request failed after multiple attempts."
            ) from last_error
        time.sleep(2**attempt)
    else:  # Defensive fallback; the loop always breaks or raises.
        raise RuntimeError("OpenAI Embeddings API request did not complete.")

    data = decoded.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError("API response contains an unexpected number of embeddings.")

    try:
        ordered = sorted(data, key=lambda item: item["index"])
        vectors = np.asarray(
            [item["embedding"] for item in ordered], dtype=np.float32
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("API response contains malformed embedding data.") from error

    if vectors.shape != (len(texts), dimensions):
        raise ValueError(
            f"Expected embedding shape {(len(texts), dimensions)}, "
            f"received {vectors.shape}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("API returned non-finite embedding values.")

    usage = decoded.get("usage", {})
    total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
    if type(total_tokens) is not int or total_tokens < 0:
        total_tokens = 0
    return vectors, total_tokens


def embed_in_batches(
    texts: list[str],
    model: str,
    dimensions: int,
    api_key: str,
    batch_size: int,
    *,
    request_function: Callable[..., tuple[np.ndarray, int]] = request_embeddings,
) -> tuple[np.ndarray, int]:
    """Embed a dataset in smaller requests and combine results in order."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    batches: list[np.ndarray] = []
    total_tokens = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start : start + batch_size]
        vectors, tokens = request_function(
            batch,
            model,
            dimensions,
            api_key,
        )
        batches.append(vectors)
        total_tokens += tokens
        print(f"Embedded API batch {batch_number}/{total_batches}.")

    if not batches:
        raise ValueError("There is no text to embed.")
    return np.vstack(batches), total_tokens
