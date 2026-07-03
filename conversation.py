"""
conversation.py — Multi-turn conversation intelligence for the Vera bot.

Handles /v1/reply by running three detectors in sequence:
  1. Auto-Reply Detector    — exits gracefully, doesn't waste turns
  2. Intent Detector        — switches to ACTION_MODE on commit signals
  3. Graceful Exit Handler  — ends on hostility/explicit opt-out

Then composes the appropriate response using the composer.
"""
from __future__ import annotations

import re
from typing import Optional

import store
from composer import compose

# ---------------------------------------------------------------------------
# Auto-reply patterns (WhatsApp Business canned responses)
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    # English patterns
    r"thank you for contact",
    r"our team will respond",
    r"will get back to you",
    r"automated (message|response|reply|assistant)",
    r"we have received your (message|query|enquiry)",
    r"we will reply (shortly|soon|within)",
    r"this is an (automated|auto) (message|reply|response)",
    r"i am (an automated|a bot|a virtual)",
    r"hi.*this is an automated",
    r"thank.*message.*shortly",
    r"team.*back.*you.*soon",
    # Hindi patterns
    r"aapki jaankari ke liye.*shukriya",
    r"main ek automated assistant",
    r"hamari team.*pahunch",
    r"jaldi.*jawab.*milega",
    r"shukriya.*sampark",
]

# ---------------------------------------------------------------------------
# Intent signals
# ---------------------------------------------------------------------------

ACTION_SIGNALS = [
    # English
    r"\b(yes|yep|yeah|yup|sure|ok|okay|alright|go ahead|let'?s do it|proceed|confirm|sounds good)\b",
    r"\b(go|start|begin|karo|chaliye|haan)\b",
    r"\blet'?s go\b",
    r"\bdo it\b",
    r"\bI('m| am) in\b",
    # Hindi
    r"\bhaan\b",
    r"\btheek hai\b",
    r"\bkaro\b",
    r"\bchaliye\b",
    r"\baage badho\b",
]

REJECTION_SIGNALS = [
    r"\b(no|nope|nahi|nope|not interested|mat karo|band karo|stop)\b",
    r"\b(don'?t|do not) (message|contact|bother|call|send)\b",
    r"\bnot now\b",
    r"\bmaybe later\b",
    r"\bkoi zarurat nahi\b",
]

HOSTILE_SIGNALS = [
    r"\b(spam|useless|stupid|idiot|moron|shut up|f.?ck|b.?tch|a.?shole)\b",
    r"\b(stop (bothering|messaging|spamming))\b",
    r"\bwhy are you (bothering|messaging|spamming)\b",
    r"\bblock you\b",
    r"\breport you\b",
    r"\bdo not (ever |)contact\b",
]

OUT_OF_SCOPE_SIGNALS = [
    r"\bgst (filing|return|payment)\b",
    r"\bincome tax\b",
    r"\bloan\b",
    r"\bjob\b",
    r"\binsurance\b",
    r"\bvisa\b",
    r"\bpassport\b",
    r"\bca \b",
    r"\bchartered accountant\b",
]


# ---------------------------------------------------------------------------
# Detector functions
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def detect_auto_reply(message: str) -> bool:
    """Return True if the message looks like a WA Business auto-reply."""
    return _matches_any(message, AUTO_REPLY_PATTERNS)


def detect_intent(message: str) -> str:
    """
    Classify the merchant's reply intent.
    Returns: "action" | "reject" | "hostile" | "out_of_scope" | "engaged"
    """
    if _matches_any(message, HOSTILE_SIGNALS):
        return "hostile"
    if _matches_any(message, REJECTION_SIGNALS):
        return "reject"
    if _matches_any(message, OUT_OF_SCOPE_SIGNALS):
        return "out_of_scope"
    if _matches_any(message, ACTION_SIGNALS):
        return "action"
    return "engaged"


# ---------------------------------------------------------------------------
# LLM-powered reply composer for genuine merchant messages
# ---------------------------------------------------------------------------

async def _compose_reply(
    conv_id: str,
    merchant_id: str,
    customer_id: Optional[str],
    message: str,
    intent: str,
    turn_number: int,
) -> dict:
    """
    Compose the bot's next reply using LLM, anchored to the conversation context.
    """
    merchant = store.get_merchant(merchant_id) or {}
    category = store.get_category_for_merchant(merchant_id) or {}
    customer = store.get_customer(customer_id) if customer_id else None
    conv = store.get_conversation(conv_id) or {}

    # Build a synthetic "reply trigger" so the composer has context
    # about why we're composing and what the merchant said
    category_slug = category.get("slug", "")
    category_voice = category.get("voice", {}).get("tone", "")
    reply_trigger = {
        "id": f"reply_{conv_id}_{turn_number}",
        "scope": "customer" if customer else "merchant",
        "kind": "active_planning_intent" if intent == "action" else "engaged_reply",
        "source": "internal",
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "payload": {
            "intent": intent,
            "merchant_message": message,
            "turn_number": turn_number,
            "intent_topic": _extract_topic(message, conv),
            "merchant_last_message": message,
            "category_slug": category_slug,
            "category_voice": category_voice,
        },
        "urgency": 4,
        "suppression_key": f"reply:{conv_id}:{turn_number}",
        "expires_at": "2099-01-01T00:00:00Z",
    }

    # For action intent, use active_planning framing
    if intent == "action":
        reply_trigger["kind"] = "active_planning_intent"

    prior_history = conv.get("turns", [])

    result = await compose(category, merchant, reply_trigger, customer, prior_history)
    return result


def _extract_topic(message: str, conv: dict) -> str:
    """Extract the conversation topic from message or conversation history."""
    # Look for topic in prior turns
    turns = conv.get("turns", [])
    for t in reversed(turns):
        body = t.get("body", "")
        if body and len(body) > 20:
            return body[:100]
    return message[:100]


# ---------------------------------------------------------------------------
# Main reply handler — called from /v1/reply endpoint
# ---------------------------------------------------------------------------

async def handle_reply(
    conversation_id: str,
    merchant_id: str,
    customer_id: Optional[str],
    message: str,
    turn_number: int,
) -> dict:
    """
    Process a reply from the merchant/customer and return the bot's next action.

    Returns one of:
      {"action": "send", "body": str, "cta": str, "rationale": str}
      {"action": "wait", "wait_seconds": int, "rationale": str}
      {"action": "end", "rationale": str}
    """
    # --- Guard: conversation already ended? ---
    if store.is_conversation_ended(conversation_id):
        return {
            "action": "end",
            "rationale": "Conversation was already ended; not re-engaging.",
        }

    # --- Append the incoming turn to history ---
    store.append_turn(conversation_id, "merchant", message)
    conv = store.get_or_create_conversation(conversation_id)

    # =========================================================
    # 1. AUTO-REPLY DETECTION
    # =========================================================
    if detect_auto_reply(message):
        auto_count = store.increment_auto_reply(merchant_id, message)

        if auto_count == 1:
            # First auto-reply: send ONE message flagging it for the owner
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this — just reply YES if you'd like to continue. STOP to opt out.",
                "cta": "binary_yes_stop",
                "rationale": "Detected WA Business auto-reply (canned response pattern). Sending one bridging message for the owner; will wait if auto-reply repeats.",
            }
        elif auto_count == 2:
            # Second consecutive auto-reply: back off for 4 hours
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": f"Auto-reply detected {auto_count}x consecutively — owner not at phone. Backing off 4 hours.",
            }
        else:
            # Third+ auto-reply: end the conversation
            store.end_conversation(conversation_id, "auto_reply_exhausted")
            return {
                "action": "end",
                "rationale": f"Auto-reply received {auto_count}x in a row. No real engagement. Closing conversation.",
            }

    # Not an auto-reply — reset the auto-reply counter
    store.reset_auto_reply(merchant_id)

    # =========================================================
    # 2. INTENT DETECTION
    # =========================================================
    intent = detect_intent(message)

    # --- Hostile message ---
    if intent == "hostile":
        store.end_conversation(conversation_id, "merchant_hostile")
        return {
            "action": "end",
            "rationale": "Merchant expressed strong frustration/hostility. Closing conversation; suppressing all triggers for this merchant for 30 days.",
        }

    # --- Explicit rejection ---
    if intent == "reject":
        store.end_conversation(conversation_id, "merchant_rejected")
        return {
            "action": "send",
            "body": "Samajh gaya — I won't message again. If you ever want to reconnect, just say 'Hi Vera'. 🙏",
            "cta": "none",
            "rationale": "Merchant explicitly opted out. Sending one graceful closing message then ending the conversation.",
        }

    # --- Out-of-scope question ---
    if intent == "out_of_scope":
        # Polite redirect — stay on-mission, don't abandon the thread
        prev_turns = conv.get("turns", [])
        last_vera_body = ""
        for t in reversed(prev_turns):
            if t.get("from") == "vera":
                last_vera_body = t.get("body", "")[:80]
                break
        redirect = f" Picking up where we left off — {last_vera_body[:50]}..." if last_vera_body else ""
        return {
            "action": "send",
            "body": f"That's outside what I handle — a CA or tax consultant would be your best bet for that. {redirect} Want to continue? Reply YES.",
            "cta": "binary_yes_stop",
            "rationale": "Merchant asked out-of-scope question. Politely declined and redirected to original conversation thread.",
        }

    # =========================================================
    # 3. GENUINE ENGAGEMENT — compose an LLM reply
    # =========================================================
    # Set conversation state
    if intent == "action":
        store.set_conversation_state(conversation_id, "action_mode")
    else:
        store.set_conversation_state(conversation_id, "engaged")

    try:
        result = await _compose_reply(
            conversation_id, merchant_id, customer_id, message, intent, turn_number
        )
        body = result.get("body", "")

        # Anti-repetition: check against previously sent messages
        prev_bodies = store.get_previous_bodies(conversation_id)
        if body in prev_bodies:
            # Add a differentiator
            body = body + "\n\n(Following up on the above — let me know how you'd like to proceed.)"

        store.append_turn(conversation_id, "vera", body)

        return {
            "action": "send",
            "body": body,
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", "Composed from full 4-context package."),
        }

    except Exception as e:
        # Safe fallback reply
        return {
            "action": "send",
            "body": "Got it! Give me a moment — I'll put together the details and send them over shortly.",
            "cta": "none",
            "rationale": f"LLM reply composition failed: {str(e)[:80]}. Sent safe acknowledgment.",
        }
