"""Validate the cleaned tweet dataset without displaying private tweet text."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "tweets_clean.json"

REQUIRED_FIELDS = (
    "id",
    "text",
    "created_at",
    "likes",
    "retweets",
    "language",
    "is_reply",
    "is_retweet",
    "reply_to_tweet_id",
    "reply_to_user_id",
    "source_file",
)

ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]"
)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]"
)


def load_dataset(input_file: Path) -> list[Any]:
    """Load the processed JSON file and confirm its top-level structure."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Processed dataset does not exist: {input_file}. "
            "Run src/parse_tweets.py first."
        )
    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")

    try:
        raw_content = input_file.read_text(encoding="utf-8")
    except OSError as error:
        raise OSError(f"Could not read {input_file}: {error}") from error

    try:
        records = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise ValueError(f"The processed dataset is not valid JSON: {error}") from error

    if not isinstance(records, list):
        raise ValueError("The processed dataset must contain a top-level JSON list.")

    return records


def is_missing(value: Any) -> bool:
    """Return whether a value should be counted as missing."""
    return value is None or (isinstance(value, str) and not value.strip())


def validate_dataset(records: list[Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate records and return an aggregate summary plus error messages."""
    errors: list[str] = []
    missing_values: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    valid_dates: list[datetime] = []
    invalid_record_types = 0
    missing_field_occurrences = 0
    invalid_id_count = 0
    empty_text_count = 0
    invalid_date_count = 0
    invalid_count_fields = 0
    invalid_boolean_fields = 0
    invalid_optional_id_fields = 0
    invalid_language_fields = 0
    invalid_source_fields = 0
    inconsistent_reply_flags = 0
    inconsistent_retweet_flags = 0
    reply_count = 0
    retweet_count = 0
    arabic_record_count = 0
    emoji_record_count = 0

    for record in records:
        if not isinstance(record, dict):
            invalid_record_types += 1
            continue

        for field in REQUIRED_FIELDS:
            if field not in record:
                missing_field_occurrences += 1
                missing_values[field] += 1
            elif is_missing(record[field]):
                missing_values[field] += 1

        tweet_id = record.get("id")
        if not isinstance(tweet_id, str) or not tweet_id.strip():
            invalid_id_count += 1
        elif tweet_id in seen_ids:
            duplicate_ids += 1
        else:
            seen_ids.add(tweet_id)

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            empty_text_count += 1
        else:
            arabic_record_count += bool(ARABIC_PATTERN.search(text))
            emoji_record_count += bool(EMOJI_PATTERN.search(text))

        created_at = record.get("created_at")
        if not isinstance(created_at, str):
            invalid_date_count += 1
        else:
            try:
                parsed_date = datetime.fromisoformat(created_at)
            except ValueError:
                invalid_date_count += 1
            else:
                if parsed_date.tzinfo is None:
                    invalid_date_count += 1
                else:
                    valid_dates.append(parsed_date)

        for field in ("likes", "retweets"):
            value = record.get(field)
            if type(value) is not int or value < 0:
                invalid_count_fields += 1

        for field in ("is_reply", "is_retweet"):
            if type(record.get(field)) is not bool:
                invalid_boolean_fields += 1

        for field in ("reply_to_tweet_id", "reply_to_user_id"):
            value = record.get(field)
            if value is not None and not isinstance(value, str):
                invalid_optional_id_fields += 1

        language = record.get("language")
        if language is not None and not isinstance(language, str):
            invalid_language_fields += 1

        source_file = record.get("source_file")
        if not isinstance(source_file, str) or not source_file.strip():
            invalid_source_fields += 1

        is_reply = record.get("is_reply")
        reply_to_tweet_id = record.get("reply_to_tweet_id")
        if type(is_reply) is bool:
            reply_count += is_reply
            if is_reply != (reply_to_tweet_id is not None):
                inconsistent_reply_flags += 1

        is_retweet = record.get("is_retweet")
        if type(is_retweet) is bool:
            retweet_count += is_retweet
            if isinstance(text, str):
                expected_retweet = text.lstrip().startswith("RT @")
                if is_retweet != expected_retweet:
                    inconsistent_retweet_flags += 1

    if invalid_record_types:
        errors.append(f"{invalid_record_types} records are not JSON objects")
    if missing_field_occurrences:
        errors.append(
            f"{missing_field_occurrences} required field occurrences are absent"
        )
    if invalid_id_count:
        errors.append(f"{invalid_id_count} records have an invalid ID")
    if duplicate_ids:
        errors.append(f"{duplicate_ids} duplicate IDs were found")
    if empty_text_count:
        errors.append(f"{empty_text_count} records have empty or invalid text")
    if invalid_date_count:
        errors.append(f"{invalid_date_count} records have an invalid date")
    if invalid_count_fields:
        errors.append(
            f"{invalid_count_fields} engagement fields are not non-negative integers"
        )
    if invalid_boolean_fields:
        errors.append(f"{invalid_boolean_fields} flag fields are not booleans")
    if invalid_optional_id_fields:
        errors.append(
            f"{invalid_optional_id_fields} optional reply IDs are not strings or null"
        )
    if invalid_language_fields:
        errors.append(
            f"{invalid_language_fields} language fields are not strings or null"
        )
    if invalid_source_fields:
        errors.append(f"{invalid_source_fields} source filenames are invalid")
    if inconsistent_reply_flags:
        errors.append(
            f"{inconsistent_reply_flags} reply flags disagree with reply tweet IDs"
        )
    if inconsistent_retweet_flags:
        errors.append(
            f"{inconsistent_retweet_flags} retweet flags disagree with tweet structure"
        )

    dates_in_order = valid_dates == sorted(valid_dates)
    if valid_dates and not dates_in_order:
        errors.append("Records are not sorted from oldest to newest")

    now = datetime.now(timezone.utc)
    earliest_date = min(valid_dates) if valid_dates else None
    latest_date = max(valid_dates) if valid_dates else None
    if earliest_date and earliest_date.year < 2006:
        errors.append("The earliest date predates Twitter/X")
    if latest_date and latest_date > now + timedelta(days=1):
        errors.append("The latest date is unexpectedly in the future")

    summary = {
        "record_count": len(records),
        "unique_id_count": len(seen_ids),
        "reply_count": reply_count,
        "retweet_count": retweet_count,
        "arabic_record_count": arabic_record_count,
        "emoji_record_count": emoji_record_count,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "dates_in_order": dates_in_order,
        "missing_values": missing_values,
    }
    return summary, errors


def format_date(value: datetime | None) -> str:
    """Format a date for the aggregate validation report."""
    return value.date().isoformat() if value else "Not available"


def print_report(
    input_file: Path, summary: dict[str, Any], errors: list[str]
) -> None:
    """Print validation results without printing tweet text or identifiers."""
    print("Clean tweet dataset validation")
    print("------------------------------")
    print(f"Input file:             {input_file}")
    print(f"Total records:          {summary['record_count']}")
    print(f"Unique IDs:             {summary['unique_id_count']}")
    print(f"Replies:                {summary['reply_count']}")
    print(f"Retweets:               {summary['retweet_count']}")
    print(f"Arabic records:         {summary['arabic_record_count']}")
    print(f"Emoji records:          {summary['emoji_record_count']}")
    print(f"Earliest date:          {format_date(summary['earliest_date'])}")
    print(f"Latest date:            {format_date(summary['latest_date'])}")
    print(f"Oldest-to-newest order: {'Yes' if summary['dates_in_order'] else 'No'}")
    print()
    print("Missing values by field")
    print("-----------------------")
    for field in REQUIRED_FIELDS:
        print(f"{field:<22} {summary['missing_values'][field]}")
    print()

    if errors:
        print("Validation failed")
        print("-----------------")
        for error in errors:
            print(f"- {error}")
    else:
        print("Validation passed: no structural or data-quality errors found.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the validator's command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate the cleaned X/Twitter tweet dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Processed JSON file to validate (default: {DEFAULT_INPUT_FILE})",
    )
    return parser


def main() -> int:
    """Run validation and return zero only when all checks pass."""
    arguments = build_argument_parser().parse_args()

    try:
        records = load_dataset(arguments.input)
        summary, errors = validate_dataset(records)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Validation error: {error}")
        return 1

    print_report(arguments.input, summary, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
