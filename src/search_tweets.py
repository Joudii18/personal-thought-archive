"""Search the cleaned tweet archive using ordinary keyword matching."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "tweets_clean.json"


def load_tweets(input_file: Path) -> list[dict[str, Any]]:
    """Load cleaned tweet records from JSON."""
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

    if not isinstance(data, list):
        raise ValueError("The clean dataset must contain a JSON list.")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError("Every item in the clean dataset must be a JSON object.")

    return data


def get_post_type(tweet: dict[str, Any]) -> str:
    """Return a readable type for one tweet record."""
    if tweet.get("is_retweet") is True:
        return "retweet"
    if tweet.get("is_reply") is True:
        return "reply"
    return "original"


def parse_created_at(tweet: dict[str, Any]) -> datetime:
    """Parse a cleaned ISO 8601 timestamp for sorting and filtering."""
    value = tweet.get("created_at")
    if not isinstance(value, str):
        raise ValueError("A tweet has a missing or invalid created_at value.")

    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"A tweet has an invalid created_at value: {value!r}") from error


def search_tweets(
    tweets: list[dict[str, Any]],
    query: str,
    *,
    year: int | None = None,
    language: str | None = None,
    post_type: str = "all",
    sort_by: str = "newest",
) -> list[dict[str, Any]]:
    """Return tweets whose text contains the query and matches all filters.

    Matching is case-insensitive and works with both English and Arabic text.
    This is lexical search: the exact query characters must occur in the text.
    """
    normalized_query = query.strip().casefold()
    if not normalized_query:
        raise ValueError("Search query cannot be empty.")

    if post_type not in {"all", "original", "reply", "retweet"}:
        raise ValueError(f"Unsupported post type: {post_type}")
    if sort_by not in {"newest", "oldest", "most-liked"}:
        raise ValueError(f"Unsupported sort option: {sort_by}")

    normalized_language = language.casefold() if language else None
    matches: list[dict[str, Any]] = []

    for tweet in tweets:
        text = tweet.get("text")
        if not isinstance(text, str) or normalized_query not in text.casefold():
            continue

        created_at = parse_created_at(tweet)
        if year is not None and created_at.year != year:
            continue

        tweet_language = tweet.get("language")
        if normalized_language is not None and (
            not isinstance(tweet_language, str)
            or tweet_language.casefold() != normalized_language
        ):
            continue

        if post_type != "all" and get_post_type(tweet) != post_type:
            continue

        matches.append(tweet)

    if sort_by == "oldest":
        return sorted(matches, key=parse_created_at)
    if sort_by == "most-liked":
        return sorted(
            matches,
            key=lambda tweet: (tweet.get("likes", 0), parse_created_at(tweet)),
            reverse=True,
        )
    return sorted(matches, key=parse_created_at, reverse=True)


def positive_integer(value: str) -> int:
    """Convert a command-line value into a positive integer."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error

    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def print_results(
    matches: list[dict[str, Any]], query: str, limit: int
) -> None:
    """Display a limited set of matches without printing tweet IDs."""
    print(f"Found {len(matches)} match(es) for {query!r}.")
    if not matches:
        return

    displayed_matches = matches[:limit]
    print(f"Showing {len(displayed_matches)} result(s):")

    for position, tweet in enumerate(displayed_matches, start=1):
        created_at = parse_created_at(tweet).date().isoformat()
        language = tweet.get("language") or "unknown"
        likes = tweet.get("likes", 0)
        retweets = tweet.get("retweets", 0)

        print()
        print(f"{position}. {created_at} | {get_post_type(tweet)} | {language}")
        print(f"   Likes: {likes} | Retweets: {retweets}")
        print(f"   {tweet['text']}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the search command's user interface."""
    parser = argparse.ArgumentParser(
        description="Search cleaned tweets using exact keyword matching."
    )
    parser.add_argument("query", help="Word or phrase to find in tweet text")
    parser.add_argument("--year", type=int, help="Only search tweets from this year")
    parser.add_argument("--language", help="Only search one archive language code")
    parser.add_argument(
        "--type",
        dest="post_type",
        choices=("all", "original", "reply", "retweet"),
        default="all",
        help="Only search one post type (default: all)",
    )
    parser.add_argument(
        "--sort",
        dest="sort_by",
        choices=("newest", "oldest", "most-liked"),
        default="newest",
        help="Result order (default: newest)",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        default=10,
        help="Maximum number of tweets to display (default: 10)",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print only the number of matches, not tweet text",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Clean JSON input file (default: {DEFAULT_INPUT_FILE})",
    )
    return parser


def main() -> int:
    """Load the dataset, search it, and print results."""
    arguments = build_argument_parser().parse_args()

    try:
        tweets = load_tweets(arguments.input)
        matches = search_tweets(
            tweets,
            arguments.query,
            year=arguments.year,
            language=arguments.language,
            post_type=arguments.post_type,
            sort_by=arguments.sort_by,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Search error: {error}")
        return 1

    if arguments.count_only:
        print(f"Found {len(matches)} match(es) for {arguments.query!r}.")
    else:
        print_results(matches, arguments.query, arguments.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
