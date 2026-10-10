"""Shared helpers for building and searching local tweet embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TWEETS_FILE = PROJECT_ROOT / "data" / "processed" / "tweets_clean.json"
DEFAULT_EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "processed" / "tweet_embeddings.npz"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / ".model-cache"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def configure_local_model_cache(cache_folder: Path) -> None:
    """Keep Hugging Face model files and logs inside the private project cache."""
    cache_folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_folder))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_folder / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_folder / "transformers"))


def load_tweets(input_file: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the cleaned tweet dataset."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {input_file}. Run src/parse_tweets.py first."
        )

    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"The clean dataset is not valid JSON: {error}") from error
    except OSError as error:
        raise OSError(f"Could not read {input_file}: {error}") from error

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("The clean dataset must be a JSON list of objects.")

    ids: set[str] = set()
    for index, tweet in enumerate(data):
        tweet_id = tweet.get("id")
        text = tweet.get("text")
        if not isinstance(tweet_id, str) or not tweet_id:
            raise ValueError(f"Tweet at position {index} has an invalid ID.")
        if tweet_id in ids:
            raise ValueError(f"Duplicate tweet ID found at position {index}.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Tweet at position {index} has invalid text.")
        ids.add(tweet_id)

    return data


def dataset_fingerprint(tweets: list[dict[str, Any]]) -> str:
    """Return a stable hash that changes if tweet IDs, text, or order changes."""
    digest = hashlib.sha256()
    for tweet in tweets:
        digest.update(tweet["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(tweet["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_embedding_index(
    output_file: Path,
    tweet_ids: list[str],
    embeddings: np.ndarray,
    model_name: str,
    fingerprint: str,
) -> None:
    """Save embeddings and the metadata needed to verify their alignment."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if matrix.shape[0] != len(tweet_ids):
        raise ValueError("Embedding count does not match tweet ID count.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        tweet_ids=np.asarray(tweet_ids),
        embeddings=matrix,
        model_name=np.asarray(model_name),
        dataset_fingerprint=np.asarray(fingerprint),
    )


def load_embedding_index(
    embeddings_file: Path,
) -> tuple[list[str], np.ndarray, str, str]:
    """Load vectors and metadata from the compressed local index."""
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embedding index not found: {embeddings_file}. "
            "Run src/build_embeddings.py first."
        )

    try:
        with np.load(embeddings_file, allow_pickle=False) as index:
            required = {
                "tweet_ids",
                "embeddings",
                "model_name",
                "dataset_fingerprint",
            }
            missing = required.difference(index.files)
            if missing:
                raise ValueError(
                    "Embedding index is missing: " + ", ".join(sorted(missing))
                )
            tweet_ids = index["tweet_ids"].astype(str).tolist()
            embeddings = index["embeddings"].astype(np.float32)
            model_name = str(index["model_name"].item())
            fingerprint = str(index["dataset_fingerprint"].item())
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load the embedding index: {error}") from error

    if embeddings.ndim != 2 or embeddings.shape[0] != len(tweet_ids):
        raise ValueError("Embedding index has inconsistent dimensions.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding index contains non-finite values.")

    return tweet_ids, embeddings, model_name, fingerprint


def verify_embedding_alignment(
    tweets: list[dict[str, Any]],
    tweet_ids: list[str],
    embeddings: np.ndarray,
    saved_fingerprint: str,
) -> None:
    """Ensure each saved vector still belongs to the corresponding tweet."""
    current_ids = [tweet["id"] for tweet in tweets]
    if current_ids != tweet_ids:
        raise ValueError(
            "Tweet IDs or ordering changed. Rebuild embeddings with "
            "src/build_embeddings.py."
        )
    if embeddings.shape[0] != len(tweets):
        raise ValueError("Tweet and embedding counts differ. Rebuild embeddings.")
    if dataset_fingerprint(tweets) != saved_fingerprint:
        raise ValueError(
            "Tweet text changed after embeddings were built. Rebuild embeddings."
        )
