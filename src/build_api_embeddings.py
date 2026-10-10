"""Build a tweet embedding index using the OpenAI Embeddings API."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .embedding_utils import (
        DEFAULT_TWEETS_FILE,
        PROJECT_ROOT,
        dataset_fingerprint,
        load_tweets,
        save_embedding_index,
    )
    from .openai_embeddings_client import (
        DEFAULT_API_MODEL,
        DEFAULT_DIMENSIONS,
        embed_in_batches,
        get_api_key,
    )
except ImportError:  # Support direct execution.
    from embedding_utils import (
        DEFAULT_TWEETS_FILE,
        PROJECT_ROOT,
        dataset_fingerprint,
        load_tweets,
        save_embedding_index,
    )
    from openai_embeddings_client import (
        DEFAULT_API_MODEL,
        DEFAULT_DIMENSIONS,
        embed_in_batches,
        get_api_key,
    )


DEFAULT_API_INDEX = PROJECT_ROOT / "data" / "processed" / "tweet_embeddings_api.npz"


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the API embedding builder's interface."""
    parser = argparse.ArgumentParser(
        description="Build tweet embeddings using the OpenAI API."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TWEETS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_API_INDEX)
    parser.add_argument("--model", default=DEFAULT_API_MODEL)
    parser.add_argument("--dimensions", type=positive_integer, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--batch-size", type=positive_integer, default=100)
    return parser


def main() -> int:
    """Send tweet batches to the API and save returned vectors locally."""
    arguments = build_argument_parser().parse_args()
    try:
        api_key = get_api_key()
        tweets = load_tweets(arguments.input)
        print(f"Loaded {len(tweets):,} cleaned tweets.")
        print(
            "Privacy notice: tweet text will be sent to the OpenAI "
            "Embeddings API."
        )
        embeddings, total_tokens = embed_in_batches(
            [tweet["text"] for tweet in tweets],
            arguments.model,
            arguments.dimensions,
            api_key,
            arguments.batch_size,
        )
        save_embedding_index(
            arguments.output,
            [tweet["id"] for tweet in tweets],
            embeddings,
            arguments.model,
            dataset_fingerprint(tweets),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"API embedding error: {error}")
        return 1

    print("API embedding index created successfully.")
    print(f"Tweets embedded:    {len(tweets):,}")
    print(f"Vector dimensions:  {embeddings.shape[1]}")
    print(f"Tokens reported:    {total_tokens:,}")
    print(f"Output file:        {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
