"""
bot.py — FastAPI application for the magicpin Vera AI bot.

Exposes 5 endpoints per the challenge-testing-brief.md spec:
  GET  /v1/healthz    — liveness probe
  GET  /v1/metadata   — bot identity
  POST /v1/context    — receive context push
  POST /v1/tick       — periodic wake-up; bot initiates proactive messages
  POST /v1/reply      — receive a reply from merchant/customer

Design principles:
  - /v1/tick must complete in <30s; uses asyncio.gather for parallel composition
  - All context reads from store.py (always latest version)
  - Suppression tracking prevents duplicate sends
  - Conversation state persisted across calls
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import store
import config
from composer import compose
from conversation import handle_reply

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="Vera — magicpin Merchant AI", version=config.BOT_VERSION)
START_TIME = time.time()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v):
        valid = {"category", "merchant", "customer", "trigger"}
        if v not in valid:
            raise ValueError(f"scope must be one of {valid}")
        return v


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": store.get_context_counts(),
    }


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": config.TEAM_NAME,
        "team_members": [config.TEAM_NAME],
        "model": config.GEMINI_MODEL,
        "approach": (
            "Trigger-kind dispatch + 4-context composer (category, merchant, trigger, customer). "
            "Gemini 2.0 Flash at temperature=0 for deterministic output. "
            "Multi-turn: auto-reply detection, intent transition, graceful exit. "
            "All context from live store — adapts to mid-test context injections automatically."
        ),
        "contact_email": config.CONTACT_EMAIL,
        "version": config.BOT_VERSION,
        "submitted_at": config.SUBMITTED_AT,
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

@app.post("/v1/context")
async def push_context(body: ContextBody):
    result = store.push_context(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
    )

    if not result["accepted"] and result.get("reason") == "stale_version":
        return JSONResponse(status_code=409, content=result)

    if not result["accepted"] and result.get("reason") == "invalid_scope":
        return JSONResponse(status_code=400, content=result)

    return result


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    For each available trigger:
    1. Skip if conversation already ended or suppression key already fired
    2. Resolve all 4 contexts
    3. Compose the message (in parallel, capped at 20 actions)
    4. Mark suppression key
    5. Return actions list
    """
    available = body.available_triggers[:20]  # cap per spec
    if not available:
        return {"actions": []}

    # Fire all compositions in parallel
    tasks = [_try_compose_for_trigger(tid) for tid in available]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    actions = []
    for result in results:
        if isinstance(result, Exception):
            # Log but don't fail the whole tick
            continue
        if result is not None:
            actions.append(result)

    return {"actions": actions}


async def _try_compose_for_trigger(trigger_id: str) -> Optional[dict]:
    """
    Resolve contexts for a trigger and compose a message.
    Returns None if the trigger should be skipped.
    """
    trigger, merchant, category, customer = store.resolve_trigger_context(trigger_id)

    # Skip if missing required contexts
    if not trigger or not merchant or not category:
        return None

    # Skip if suppression key already fired
    suppression_key = trigger.get("suppression_key", "")
    if suppression_key and store.is_suppressed(suppression_key):
        return None

    # Generate a stable conversation_id for this trigger
    merchant_id = trigger.get("merchant_id", "")
    customer_id = trigger.get("customer_id")
    trigger_kind = trigger.get("kind", "general")
    conv_id = f"conv_{merchant_id}_{trigger_id}".replace(" ", "_")

    # Skip if this conversation was already ended
    if store.is_conversation_ended(conv_id):
        return None

    # Get conversation history from merchant context (already in merchant payload)
    conv_history = merchant.get("conversation_history", [])

    # Compose the message
    try:
        result = await compose(category, merchant, trigger, customer, conv_history)
    except Exception as e:
        return None

    body = result.get("body", "")
    if not body:
        return None

    # Anti-repetition: don't send the same body twice in this conversation
    prev_bodies = store.get_previous_bodies(conv_id)
    if body in prev_bodies:
        return None

    # Mark suppression key as fired
    if suppression_key:
        store.mark_suppressed(suppression_key)

    # Record this outbound turn in conversation state
    store.append_turn(conv_id, "vera", body)

    # Determine send_as
    send_as = result.get("send_as", "vera")

    # Build the action object
    owner = merchant.get("identity", {}).get("owner_first_name", "") or merchant.get("identity", {}).get("name", "")
    template_name = result.get("template_name", f"vera_{trigger_kind}_v1")
    template_params = result.get("template_params", [owner, trigger_kind, ""])

    action = {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": send_as,
        "trigger_id": trigger_id,
        "template_name": template_name,
        "template_params": template_params[:3],  # max 3 params
        "body": body,
        "cta": result.get("cta", "open_ended"),
        "suppression_key": suppression_key,
        "rationale": result.get("rationale", ""),
    }

    return action


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Handle a reply from the merchant/customer.
    Routes through conversation.py for multi-turn intelligence.
    """
    result = await handle_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id or "",
        customer_id=body.customer_id,
        message=body.message,
        turn_number=body.turn_number,
    )

    # Record bot's response in conversation if it's a send
    if result.get("action") == "send":
        store.append_turn(body.conversation_id, "vera", result.get("body", ""))

    return result


# ---------------------------------------------------------------------------
# POST /v1/teardown (optional — wipe state at end of test)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    """Wipe all state at end of test."""
    store.wipe_all()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Vera bot is running. Endpoints: /v1/healthz, /v1/metadata, /v1/context, /v1/tick, /v1/reply"}
