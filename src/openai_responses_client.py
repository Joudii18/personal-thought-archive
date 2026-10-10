"""Minimal OpenAI Responses API client using Python's standard library."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_ANSWER_MODEL = "gpt-5-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 3000


def extract_output_text(response_data: dict[str, Any]) -> str:
    """Combine every output_text content block in a Responses API result."""
    top_level_text = response_data.get("output_text")
    if isinstance(top_level_text, str) and top_level_text.strip():
        return top_level_text.strip()

    pieces: list[str] = []
    output = response_data.get("output", [])
    if not isinstance(output, list):
        raise ValueError("Responses API output has an unexpected format.")

    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "output_text"
                and isinstance(block.get("text"), str)
            ):
                pieces.append(block["text"])

    answer = "\n".join(piece.strip() for piece in pieces if piece.strip())
    if not answer:
        status = response_data.get("status", "unknown")
        incomplete_details = response_data.get("incomplete_details")
        reason = (
            incomplete_details.get("reason")
            if isinstance(incomplete_details, dict)
            else None
        )
        error = response_data.get("error")
        error_message = error.get("message") if isinstance(error, dict) else None

        detail = f"status={status}"
        if reason:
            detail += f", reason={reason}"
        if error_message:
            detail += f", API message={error_message}"
        raise ValueError(f"Responses API returned no answer text ({detail}).")
    return answer


def request_answer(
    *,
    instructions: str,
    user_input: str,
    model: str,
    api_key: str,
    timeout: float = 120,
    max_attempts: int = 4,
) -> str:
    """Request one grounded text answer from the Responses API."""
    payload = json.dumps(
        {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "store": False,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        RESPONSES_URL,
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
            if not isinstance(decoded, dict):
                raise ValueError("Responses API returned an unexpected result.")
            return extract_output_text(decoded)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"OpenAI Responses API returned HTTP {error.code}."
                ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error

        if attempt + 1 == max_attempts:
            raise RuntimeError(
                "OpenAI Responses API request failed after multiple attempts."
            ) from last_error
        time.sleep(2**attempt)

    raise RuntimeError("OpenAI Responses API request did not complete.")
