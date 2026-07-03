"""
llm_client.py — Async Gemini API client for the Vera bot.

Uses the Gemini REST API directly (no SDK dependency) for minimal footprint.
Handles retries, timeouts, and JSON extraction from LLM responses.
"""
from __future__ import annotations

import json
import re
import httpx

from config import GEMINI_API_KEY, GEMINI_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)
TIMEOUT = 25.0  # seconds — stay safely under the 30s judge deadline
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Core async call
# ---------------------------------------------------------------------------

async def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """
    Call Gemini with a system + user prompt.
    Returns the raw text response.
    Raises RuntimeError on all failures after retries.
    """
    # Gemini doesn't have a formal "system" role in the REST API;
    # we prepend it to the user prompt with clear demarcation.
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,          # deterministic
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",  # request JSON mode
        },
        "safetySettings": [
            # Disable content filters that might block legitimate merchant content
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ],
    }

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(GEMINI_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except httpx.TimeoutException as e:
            last_error = e
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gemini timeout after {MAX_RETRIES + 1} attempts: {e}") from e
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code in (429, 503) and attempt < MAX_RETRIES:
                import asyncio
                # 429 = quota hit. Wait progressively: 4s then 10s.
                # With Semaphore(3) on the caller side, bursts are limited,
                # so a short-ish wait is usually enough to recover.
                wait = [4, 10][min(attempt, 1)]
                await asyncio.sleep(wait)
                continue
            raise RuntimeError(f"Gemini HTTP error {e.response.status_code}: {e.response.text[:200]}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini unexpected error: {e}") from e

    raise RuntimeError(f"Gemini failed after all retries: {last_error}")


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Extract a JSON object from LLM response text.
    Handles cases where the LLM wraps JSON in markdown code fences.
    """
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")
