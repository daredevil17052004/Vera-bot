"""
store.py — In-memory context store for the Vera bot.

Holds all 4 context types (category, merchant, customer, trigger) with
version-aware upsert, cross-reference indexes for fast lookup, and
conversation state tracking.

Design principles:
- Always read the LATEST version when composing (handles mid-test context injections)
- All operations are plain dict mutations — safe in asyncio's single-threaded event loop
- No threading.Lock: asyncio is cooperative, not preemptive — dict ops are atomic
- Conversation state is separate from context state (different lifecycle)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Core context storage
# (scope, context_id) → {"version": int, "payload": dict}
# ---------------------------------------------------------------------------
_contexts: dict[tuple[str, str], dict] = {}

# ---------------------------------------------------------------------------
# Cross-reference indexes (built/updated on every context push)
# ---------------------------------------------------------------------------
# merchant_id → category_slug
_merchant_to_category: dict[str, str] = {}

# category_slug → list of merchant_ids
_category_to_merchants: dict[str, list[str]] = {}

# trigger_id → merchant_id (and optional customer_id)
_trigger_to_merchant: dict[str, str] = {}
_trigger_to_customer: dict[str, Optional[str]] = {}

# merchant_id → list of trigger_ids targeting that merchant
_merchant_to_triggers: dict[str, list[str]] = {}

# customer_id → merchant_id
_customer_to_merchant: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Conversation state
# conversation_id → ConversationState dict
# ---------------------------------------------------------------------------
_conversations: dict[str, dict] = {}

# merchant_id → {"count": int, "text": str}
_merchant_auto_replies: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Suppression registry: suppression_key → True (already sent)
# ---------------------------------------------------------------------------
_suppressed: set[str] = set()

# ---------------------------------------------------------------------------
# Scope counters for /v1/healthz
# ---------------------------------------------------------------------------
_counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}


# ===========================================================================
# Context push / retrieval
# ===========================================================================

def push_context(scope: str, context_id: str, version: int, payload: dict) -> dict:
    """
    Idempotent upsert with version gating.

    Returns:
        {"accepted": True, "ack_id": str, "stored_at": str}  — on success
        {"accepted": False, "reason": "stale_version", "current_version": int}  — on stale
        {"accepted": False, "reason": "invalid_scope"}  — on bad scope
    """
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope", "details": f"Must be one of {valid_scopes}"}

    key = (scope, context_id)
    stored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    existing = _contexts.get(key)
    if existing and existing["version"] >= version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": existing["version"],
        }

    is_new = existing is None
    _contexts[key] = {"version": version, "payload": payload}

    # Update counts only for new entries (not version bumps)
    if is_new:
        _counts[scope] = _counts.get(scope, 0) + 1

    # Rebuild cross-reference indexes
    _update_indexes(scope, context_id, payload)

    return {
        "accepted": True,
        "ack_id": f"ack_{context_id}_v{version}",
        "stored_at": stored_at,
    }


def _update_indexes(scope: str, context_id: str, payload: dict) -> None:
    """Update all cross-reference indexes."""
    if scope == "merchant":
        mid = payload.get("merchant_id", context_id)
        cat_slug = payload.get("category_slug", "")
        _merchant_to_category[mid] = cat_slug
        if cat_slug not in _category_to_merchants:
            _category_to_merchants[cat_slug] = []
        if mid not in _category_to_merchants[cat_slug]:
            _category_to_merchants[cat_slug].append(mid)

    elif scope == "trigger":
        tid = payload.get("id", context_id)
        mid = payload.get("merchant_id", "")
        cid = payload.get("customer_id")
        _trigger_to_merchant[tid] = mid
        _trigger_to_customer[tid] = cid
        if mid not in _merchant_to_triggers:
            _merchant_to_triggers[mid] = []
        if tid not in _merchant_to_triggers[mid]:
            _merchant_to_triggers[mid].append(tid)

    elif scope == "customer":
        cid = payload.get("customer_id", context_id)
        mid = payload.get("merchant_id", "")
        _customer_to_merchant[cid] = mid


def get_payload(scope: str, context_id: str) -> Optional[dict]:
    """Retrieve the latest payload for a given scope+id, or None."""
    entry = _contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def get_category_for_merchant(merchant_id: str) -> Optional[dict]:
    """Convenience: resolve merchant → category payload."""
    cat_slug = _merchant_to_category.get(merchant_id)
    if not cat_slug:
        return None
    return get_payload("category", cat_slug)


def get_merchant(merchant_id: str) -> Optional[dict]:
    return get_payload("merchant", merchant_id)


def get_customer(customer_id: str) -> Optional[dict]:
    if not customer_id:
        return None
    return get_payload("customer", customer_id)


def get_trigger(trigger_id: str) -> Optional[dict]:
    return get_payload("trigger", trigger_id)


def resolve_trigger_context(trigger_id: str) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    """
    For a trigger_id, return (trigger, merchant, category, customer) payloads.
    Any may be None if not yet pushed.
    """
    trigger = get_trigger(trigger_id)
    if not trigger:
        return None, None, None, None

    mid = trigger.get("merchant_id") or _trigger_to_merchant.get(trigger_id)
    cid = trigger.get("customer_id") or _trigger_to_customer.get(trigger_id)

    merchant = get_merchant(mid) if mid else None
    category = get_category_for_merchant(mid) if mid else None
    customer = get_customer(cid) if cid else None

    return trigger, merchant, category, customer


def get_context_counts() -> dict:
    return dict(_counts)


# ===========================================================================
# Suppression registry
# ===========================================================================

def is_suppressed(suppression_key: str) -> bool:
    return suppression_key in _suppressed


def mark_suppressed(suppression_key: str) -> None:
    _suppressed.add(suppression_key)


# ===========================================================================
# Conversation state management
# ===========================================================================

def _init_conversation(conversation_id: str) -> dict:
    return {
        "id": conversation_id,
        "turns": [],               # list of {"from": str, "body": str, "ts": str}
        "ended": False,            # if True, never send again
        "auto_reply_count": 0,     # consecutive auto-reply turns
        "last_auto_reply_text": None,
        "state": "initiated",      # initiated | engaged | action_mode | qualifying | waiting | ended
        "merchant_id": None,
        "customer_id": None,
    }


def get_or_create_conversation(conversation_id: str) -> dict:
    """Return existing state or initialise a fresh one."""
    if conversation_id not in _conversations:
        _conversations[conversation_id] = _init_conversation(conversation_id)
    return _conversations[conversation_id]


def append_turn(conversation_id: str, role: str, body: str) -> None:
    """Append a turn to the conversation history."""
    conv = get_or_create_conversation(conversation_id)
    conv["turns"].append({
        "from": role,
        "body": body,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def get_conversation(conversation_id: str) -> Optional[dict]:
    return _conversations.get(conversation_id)


def is_conversation_ended(conversation_id: str) -> bool:
    conv = _conversations.get(conversation_id)
    return conv["ended"] if conv else False


def end_conversation(conversation_id: str, reason: str = "") -> None:
    conv = get_or_create_conversation(conversation_id)
    conv["ended"] = True
    conv["state"] = "ended"
    conv["end_reason"] = reason


def set_conversation_state(conversation_id: str, state: str) -> None:
    conv = get_or_create_conversation(conversation_id)
    conv["state"] = state


def get_previous_bodies(conversation_id: str) -> list[str]:
    """Return all previously sent bot message bodies for anti-repetition checks."""
    conv = _conversations.get(conversation_id)
    if not conv:
        return []
    return [t["body"] for t in conv["turns"] if t["from"] == "vera"]


def increment_auto_reply(merchant_id: str, text: str) -> int:
    """Track consecutive auto-reply occurrences per merchant. Returns new count."""
    state = _merchant_auto_replies.setdefault(merchant_id, {"count": 0, "text": None})
    if state["text"] == text:
        state["count"] += 1
    else:
        state["count"] = 1
        state["text"] = text
    return state["count"]


def reset_auto_reply(merchant_id: str) -> None:
    if merchant_id in _merchant_auto_replies:
        _merchant_auto_replies[merchant_id] = {"count": 0, "text": None}


def wipe_all() -> None:
    """Wipe all state (for /v1/teardown)."""
    _contexts.clear()
    _merchant_to_category.clear()
    _category_to_merchants.clear()
    _trigger_to_merchant.clear()
    _trigger_to_customer.clear()
    _merchant_to_triggers.clear()
    _customer_to_merchant.clear()
    _conversations.clear()
    _suppressed.clear()
    _merchant_auto_replies.clear()
    _counts.update({"category": 0, "merchant": 0, "customer": 0, "trigger": 0})
