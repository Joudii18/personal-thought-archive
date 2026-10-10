"""Search a private tweet archive by meaning using local embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .embedding_utils import (
        DEFAULT_EMBEDDINGS_FILE,
        DEFAULT_MODEL_CACHE,
        DEFAULT_TWEETS_FILE,
        configure_local_model_cache,
        load_embedding_index,
        load_tweets,
        verify_embedding_alignment,
    )
    from .search_tweets import get_post_type, parse_created_at, positive_integer
except ImportError:  # Support running this file directly with Python.
    from embedding_utils import (
        DEFAULT_EMBEDDINGS_FILE,
        DEFAULT_MODEL_CACHE,
        DEFAULT_TWEETS_FILE,
        configure_local_model_cache,
        load_embedding_index,
        load_tweets,
        verify_embedding_alignment,
    )
    from search_tweets import get_post_type, parse_created_at, positive_integer


def load_model(model_name: str, cache_folder: Path) -> Any:
    """Load the exact local model used to build the embedding index."""
    configure_local_model_cache(cache_folder)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed. Activate .venv and run "
            "python -m pip install -r requirements.txt."
        ) from error
    return SentenceTransformer(
        model_name,
        cache_folder=str(cache_folder),
        local_files_only=True,
    )


def cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between one query and every tweet vector."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError("Query and tweet embedding dimensions do not match.")

    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0 or np.any(row_norms == 0):
        raise ValueError("A zero-length embedding cannot be compared.")
    return (matrix @ query) / (row_norms * query_norm)


def semantic_rank(
    tweets: list[dict[str, Any]],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    year: int | None = None,
    language: str | None = None,
    post_type: str = "all",
    authored_only: bool = False,
) -> list[tuple[dict[str, Any], float]]:
    """Filter tweets, calculate cosine scores, and rank highest first."""
    if len(tweets) != embeddings.shape[0]:
        raise ValueError("Tweet and embedding counts do not match.")
    if post_type not in {"all", "original", "reply", "retweet"}:
        raise ValueError(f"Unsupported post type: {post_type}")

    scores = cosine_scores(query_embedding, embeddings)
    normalized_language = language.casefold() if language else None
    ranked: list[tuple[dict[str, Any], float]] = []

    for tweet, score in zip(tweets, scores, strict=True):
        if year is not None and parse_created_at(tweet).year != year:
            continue
        tweet_language = tweet.get("language")
        if normalized_language is not None and (
            not isinstance(tweet_language, str)
            or tweet_language.casefold() != normalized_language
        ):
            continue
        if post_type != "all" and get_post_type(tweet) != post_type:
            continue
        if authored_only and get_post_type(tweet) == "retweet":
            continue
        ranked.append((tweet, float(score)))

    return sorted(ranked, key=lambda item: item[1], reverse=True)


def print_results(
    ranked: list[tuple[dict[str, Any], float]], limit: int, show_text: bool = True
) -> None:
    """Print semantic matches without exposing tweet IDs."""
    displayed = ranked[:limit]
    print(f"Showing {len(displayed)} semantic result(s):")
    for position, (tweet, score) in enumerate(displayed, start=1):
        date = parse_created_at(tweet).date().isoformat()
        language = tweet.get("language") or "unknown"
        print()
        print(
            f"{position}. Similarity {score:.3f} | {date} | "
            f"{get_post_type(tweet)} | {language}"
        )
        print(f"   Likes: {tweet.get('likes', 0)} | Retweets: {tweet.get('retweets', 0)}")
        if show_text:
            print(f"   {tweet['text']}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the semantic search command's interface."""
    parser = argparse.ArgumentParser(
        description="Search cleaned tweets by semantic similarity."
    )
    parser.add_argument("query", help="Natural-language idea to search for")
    parser.add_argument("--year", type=int)
    parser.add_argument("--language")
    parser.add_argument(
        "--type",
        dest="post_type",
        choices=("all", "original", "reply", "retweet"),
        default="all",
    )
    parser.add_argument("--limit", type=positive_integer, default=10)
    parser.add_argument("--input", type=Path, default=DEFAULT_TWEETS_FILE)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument(
        "--hide-text",
        action="store_true",
        help="Show scores and metadata without displaying tweet text",
    )
    return parser


def main() -> int:
    """Embed a query locally and rank saved tweet embeddings."""
    arguments = build_argument_parser().parse_args()
    query = arguments.query.strip()
    if not query:
        print("Semantic search error: Search query cannot be empty.")
        return 1

    try:
        tweets = load_tweets(arguments.input)
        tweet_ids, embeddings, model_name, fingerprint = load_embedding_index(
            arguments.embeddings
        )
        verify_embedding_alignment(tweets, tweet_ids, embeddings, fingerprint)
        model = load_model(model_name, arguments.model_cache)
        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        ranked = semantic_rank(
            tweets,
            embeddings,
            query_embedding,
            year=arguments.year,
            language=arguments.language,
            post_type=arguments.post_type,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Semantic search error: {error}")
        return 1

    print_results(ranked, arguments.limit, show_text=not arguments.hide_text)
    print()
    print("Similarity scores rank results; they are not probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
