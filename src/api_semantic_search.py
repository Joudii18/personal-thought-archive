"""Search tweet embeddings with API-generated query vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .build_api_embeddings import DEFAULT_API_INDEX
    from .embedding_utils import (
        DEFAULT_TWEETS_FILE,
        load_embedding_index,
        load_tweets,
        verify_embedding_alignment,
    )
    from .openai_embeddings_client import get_api_key, request_embeddings
    from .search_tweets import positive_integer
    from .semantic_search import print_results, semantic_rank
except ImportError:  # Support direct execution.
    from build_api_embeddings import DEFAULT_API_INDEX
    from embedding_utils import (
        DEFAULT_TWEETS_FILE,
        load_embedding_index,
        load_tweets,
        verify_embedding_alignment,
    )
    from openai_embeddings_client import get_api_key, request_embeddings
    from search_tweets import positive_integer
    from semantic_search import print_results, semantic_rank


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the API semantic-search interface."""
    parser = argparse.ArgumentParser(
        description="Search API-generated tweet embeddings by meaning."
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
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_API_INDEX)
    parser.add_argument("--hide-text", action="store_true")
    return parser


def main() -> int:
    """Embed one query through the API and rank the saved tweet vectors."""
    arguments = build_argument_parser().parse_args()
    query = arguments.query.strip()
    if not query:
        print("API semantic search error: Search query cannot be empty.")
        return 1

    try:
        api_key = get_api_key()
        tweets = load_tweets(arguments.input)
        ids, embeddings, model, fingerprint = load_embedding_index(
            arguments.embeddings
        )
        verify_embedding_alignment(tweets, ids, embeddings, fingerprint)
        query_vectors, _ = request_embeddings(
            [query], model, embeddings.shape[1], api_key
        )
        ranked = semantic_rank(
            tweets,
            embeddings,
            query_vectors[0],
            year=arguments.year,
            language=arguments.language,
            post_type=arguments.post_type,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"API semantic search error: {error}")
        return 1

    print_results(ranked, arguments.limit, show_text=not arguments.hide_text)
    print()
    print("Similarity scores rank results; they are not probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
