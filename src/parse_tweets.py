"""Convert X/Twitter archive tweet files into a clean JSON dataset.

The parser reads files from ``data/raw`` without modifying them and writes the
cleaned records to ``data/processed/tweets_clean.json``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "tweets_clean.json"
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def find_tweet_files(input_dir: Path) -> list[Path]:
    """Return likely tweet archive files in a stable order."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    candidates = {
        *input_dir.glob("tweet*.js"),
        *input_dir.glob("tweets*.js"),
    }
    tweet_files = sorted(path for path in candidates if path.is_file())

    if not tweet_files:
        raise FileNotFoundError(
            f"No tweet archive files were found in {input_dir}. "
            "Expected a name such as tweets.js or tweets-part1.js."
        )

    return tweet_files


def load_archive_array(file_path: Path) -> list[Any]:
    """Decode the JSON array inside a JavaScript-wrapped archive file."""
    try:
        raw_content = file_path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise OSError(f"Could not read {file_path}: {error}") from error

    array_start = raw_content.find("[")
    if array_start == -1:
        raise ValueError(f"No JSON array was found in {file_path.name}.")

    try:
        decoded, end_position = json.JSONDecoder().raw_decode(raw_content[array_start:])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Could not parse the tweet data in {file_path.name}: {error}"
        ) from error

    if not isinstance(decoded, list):
        raise ValueError(f"The top-level data in {file_path.name} is not a list.")

    trailing_content = raw_content[array_start + end_position :].strip()
    if trailing_content not in {"", ";"}:
        raise ValueError(
            f"Unexpected content appears after the tweet array in {file_path.name}."
        )

    return decoded


def parse_count(value: Any) -> tuple[int, bool]:
    """Convert an engagement count to an integer and report invalid values."""
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 0, True

    if count < 0:
        return 0, True
    return count, False


def parse_date(value: Any) -> str:
    """Convert Twitter's date format to an ISO 8601 timestamp."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("tweet date is missing")

    try:
        parsed_date = datetime.strptime(value, TWITTER_DATE_FORMAT)
    except ValueError as error:
        raise ValueError(f"tweet date has an unexpected format: {value!r}") from error

    return parsed_date.isoformat()


def clean_tweet(tweet: dict[str, Any], source_file: str) -> tuple[dict[str, Any] | None, int]:
    """Create one cleaned tweet record.

    Returns the cleaned record, or ``None`` when the source record lacks an ID
    or meaningful text. The second return value counts engagement fields that
    had to be replaced with zero because their values were invalid.
    """
    tweet_id = tweet.get("id_str") or tweet.get("id")
    text = tweet.get("full_text")

    if tweet_id is None or not str(tweet_id).strip():
        return None, 0
    if not isinstance(text, str) or not text.strip():
        return None, 0

    likes, invalid_likes = parse_count(tweet.get("favorite_count"))
    retweets, invalid_retweets = parse_count(tweet.get("retweet_count"))
    reply_to_tweet_id = (
        tweet.get("in_reply_to_status_id_str")
        or tweet.get("in_reply_to_status_id")
    )
    reply_to_user_id = (
        tweet.get("in_reply_to_user_id_str")
        or tweet.get("in_reply_to_user_id")
    )

    cleaned = {
        "id": str(tweet_id),
        "text": text.strip(),
        "created_at": parse_date(tweet.get("created_at")),
        "likes": likes,
        "retweets": retweets,
        "language": tweet.get("lang"),
        "is_reply": reply_to_tweet_id is not None,
        "is_retweet": text.lstrip().startswith("RT @"),
        "reply_to_tweet_id": (
            str(reply_to_tweet_id) if reply_to_tweet_id is not None else None
        ),
        "reply_to_user_id": (
            str(reply_to_user_id) if reply_to_user_id is not None else None
        ),
        "source_file": source_file,
    }
    return cleaned, invalid_likes + invalid_retweets


def parse_archives(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read, clean, combine, and deduplicate all tweet archive files."""
    cleaned_tweets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    report = {
        "files_processed": 0,
        "raw_records": 0,
        "exported_records": 0,
        "empty_or_invalid_records_skipped": 0,
        "duplicate_ids_removed": 0,
        "invalid_counts_defaulted": 0,
    }

    for file_path in find_tweet_files(input_dir):
        archive_items = load_archive_array(file_path)
        report["files_processed"] += 1
        report["raw_records"] += len(archive_items)

        for item in archive_items:
            if not isinstance(item, dict) or not isinstance(item.get("tweet"), dict):
                report["empty_or_invalid_records_skipped"] += 1
                continue

            try:
                cleaned, invalid_counts = clean_tweet(item["tweet"], file_path.name)
            except ValueError as error:
                raise ValueError(f"Invalid record in {file_path.name}: {error}") from error

            report["invalid_counts_defaulted"] += invalid_counts
            if cleaned is None:
                report["empty_or_invalid_records_skipped"] += 1
                continue

            if cleaned["id"] in seen_ids:
                report["duplicate_ids_removed"] += 1
                continue

            seen_ids.add(cleaned["id"])
            cleaned_tweets.append(cleaned)

    cleaned_tweets.sort(key=lambda tweet: tweet["created_at"])
    report["exported_records"] = len(cleaned_tweets)
    return cleaned_tweets, report


def save_cleaned_tweets(tweets: list[dict[str, Any]], output_file: Path) -> None:
    """Write cleaned tweets as readable UTF-8 JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_file.write_text(
            json.dumps(tweets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(f"Could not write {output_file}: {error}") from error


def print_report(report: dict[str, int], output_file: Path) -> None:
    """Print aggregate processing results without exposing tweet text."""
    print("Tweet archive processing report")
    print("--------------------------------")
    print(f"Files processed:                 {report['files_processed']}")
    print(f"Raw records found:               {report['raw_records']}")
    print(f"Clean records exported:          {report['exported_records']}")
    print(
        "Empty/invalid records skipped:  "
        f"{report['empty_or_invalid_records_skipped']}"
    )
    print(f"Duplicate IDs removed:           {report['duplicate_ids_removed']}")
    print(f"Invalid counts replaced with 0:  {report['invalid_counts_defaulted']}")
    print(f"Output file:                     {output_file}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the parser."""
    parser = argparse.ArgumentParser(
        description="Convert X/Twitter archive files into clean JSON."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing tweet archive files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Clean JSON output path (default: {DEFAULT_OUTPUT_FILE})",
    )
    return parser


def main() -> int:
    """Run the parser and return a process exit code."""
    arguments = build_argument_parser().parse_args()

    try:
        tweets, report = parse_archives(arguments.input_dir)
        save_cleaned_tweets(tweets, arguments.output)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    print_report(report, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
