"""
composer.py — The core LLM-powered message composition engine for Vera.

Architecture:
  compose(category, merchant, trigger, customer?) → ComposedMessage

Design:
  1. Trigger-kind dispatch: each trigger kind gets a specialized prompt
     framing that sets the *why now* anchor before the LLM runs.
  2. Master system prompt: encodes all voice rules, anti-patterns,
     compulsion levers, and output schema.
  3. Post-LLM validation: checks for taboo words, empty body,
     repetition, schema compliance. Re-prompts once on failure.
  4. Fallback: if LLM fails entirely, returns a safe minimal message
     grounded in the merchant's name and trigger kind.

Every fact in the output must trace to a field in one of the 4 contexts.
No fabrication. No hallucination. Context-grounded always.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from llm_client import call_gemini, extract_json

# ---------------------------------------------------------------------------
# MASTER SYSTEM PROMPT
# Injected for every composition regardless of trigger kind.
# ---------------------------------------------------------------------------

MASTER_SYSTEM = """You are Vera, magicpin's merchant AI assistant on WhatsApp. Your job is to compose ONE high-quality outbound message based on the 4 context blocks provided.

## CRITICAL RULES (violation = score 0)
1. NEVER fabricate data. Every number, date, source, name must come from the provided contexts.
2. NEVER use taboo words from category.voice.vocab_taboo.
3. NEVER send the same body text that was sent before in this conversation (check conversation_history).
4. NEVER include URLs (Meta will reject them).
5. ONE primary CTA only — binary YES/STOP or open-ended, never multi-choice (booking flows may use Reply 1/2).
6. Bury nothing — the CTA must be the last sentence.
7. No long preambles ("I hope you're doing well", "I'm reaching out today to").
8. Do NOT re-introduce yourself after the first message (check conversation_history).
9. Match the merchant's language preference exactly (hi-en mix → Hindi-English code-mix; en → English).

## SIGNAL PRIORITY (combine in this order for maximum decision quality)
1. TRIGGER payload — what happened? This is the "why now" anchor. Start here.
2. MERCHANT performance delta (7d_delta) — what's at stake? This is the urgency.
3. CATEGORY digest/trend signals — what's the proof? This is the credibility.
4. ACTIVE OFFERS — what's the hook? This closes the loop with a real action.
5. CUSTOMER context — who is this for? This personalizes at the individual level.
Your message MUST weave signals from at least 3 of these 5 layers.

## SPECIFICITY MANDATE (mandatory — no exceptions)
Your message body MUST contain at least 2 specific numbers drawn directly from the contexts.
Valid numbers: view counts, CTR %, call counts, offer prices (₹), review counts, delta %, days since event, peer benchmarks, customer counts, distances (km).
Use the ⚡ PRE-COMPUTED DEMAND SIGNALS block — these are ready to use verbatim.
If your draft contains zero numbers, rewrite it.

## VOICE RULES BY CATEGORY
- dentists: peer_clinical — technical terms OK (fluoride varnish, caries, IOPA), legal taboos (cure, guaranteed), cite sources (JIDA p.14 style), Dr. prefix always
- salons: warm_practical — approachable expert, emoji OK (1-2 max), mention stylist names if available
- restaurants: warm_busy_practical — fellow operator register, use "covers", "AOV", "table turnover"
- gyms: energetic_disciplined — coach-to-operator, use fitness vocabulary, no guilt-tripping members
- pharmacies: trustworthy_precise — full molecule names, batch numbers, never "miracle cure"

## COMPULSION LEVERS (use 2+ per message — cite which ones in rationale)
1. Specificity/verifiability — numbers, dates, source citations
2. Loss aversion — "you're missing X" / "before this window closes"
3. Social proof — "3 dentists in your locality did Y this month"
4. Effort externalization — "I've already drafted X — just say YES"
5. Curiosity — "want to see who?" / "want the full breakdown?"
6. Reciprocity — "I noticed Y, thought you'd want to know"
7. Asking the merchant — low-stakes question that gets them talking
8. Single binary commitment — Reply YES / STOP, not multi-choice

## WHAT FULL MARKS LOOKS LIKE (study these — then apply to the contexts below)

Dentist example:
WEAK: "Hi Doctor, want to run a discount campaign today?"
STRONG: "Dr. Rajan, your CTR dropped 31% this week (0.021 → 0.014) — peer median in South Delhi is 0.030. Your Dental Cleaning @ ₹299 is live. Should I push it to the 2,410 people who viewed you this month? Reply YES."
→ Uses: trigger (perf_dip), merchant CTR delta, peer benchmark, active offer price, view count. 5 context layers. 4 numbers.

Salon example:
WEAK: "Hi Priya, want to attract more customers?"
STRONG: "Hi Priya! 👋 Your gel nails enquiries jumped 22% this week — but bookings didn't move. Your ₹800 Gel Manicure is live on your profile. Should I draft a 'book today' WhatsApp for your top 30 returning customers? Reply YES."
→ Uses: trigger (perf_spike), offer price, customer aggregate. 3 layers. 2 numbers.

Pharmacy example:
WEAK: "Hi, there's a product update."
STRONG: "Atorvastatin 10mg (batch B/T/3310, Sun Pharma) — 14 patients on your chronic list are affected by this recall. CDSCO deadline: July 15. Should I draft their replacement notification + the Form 57? Reply YES."
→ Uses: trigger (supply_alert), molecule name, batch number, patient count, regulatory deadline. All from context.

Gym example:
WEAK: "Hey, want to run a retention campaign?"
STRONG: "Akash — 47 members haven't checked in for 15+ days. Every Hyderabad gym sees -22% in summer — your 312 total members are right on trend. Want me to draft a 'summer survival' retention WhatsApp for those 47? Reply YES."
→ Uses: trigger (seasonal_perf_dip), exact lapsed member count, city name, industry benchmark. 3 numbers.

Restaurant example:
WEAK: "Want to boost sales?"
STRONG: "Quick one — your AOV dropped ₹45 this week (₹285 → ₹240). 3 competitors added lunch combos at ₹149. Your Thali @ ₹149 isn't featured today. Should I push it to your 1,200 monthly viewers? Reply YES."
→ Uses: trigger (perf_dip), AOV delta, competitor count, offer price, view count. 5 numbers.

The difference: EVERY claim traces to a specific field in the contexts below. No claim without a source.

## OUTPUT FORMAT (respond ONLY with this JSON, no other text)
{
  "body": "<the WhatsApp message body — concise, punchy, context-grounded>",
  "cta": "<one of: binary_yes_stop | binary_yes_no | binary_confirm_cancel | open_ended | multi_choice_slot | none>",
  "send_as": "<vera | merchant_on_behalf>",
  "suppression_key": "<copy exactly from trigger.suppression_key>",
  "rationale": "<2-3 sentences: (1) which context fields drove each element, (2) which 2+ compulsion levers used, (3) which 3+ signal layers combined>",
  "template_name": "<snake_case template name for this trigger kind>",
  "template_params": ["<param1>", "<param2>", "<param3>"]
}

## send_as RULE
- If trigger.scope == "customer" → send_as = "merchant_on_behalf"
- If trigger.scope == "merchant" → send_as = "vera"
"""

# ---------------------------------------------------------------------------
# TRIGGER-KIND SPECIFIC FRAMING
# Prepended to the user prompt to set context for the LLM.
# ---------------------------------------------------------------------------

TRIGGER_FRAMING: dict[str, str] = {
    "research_digest": """TRIGGER TYPE: Research/Knowledge Digest
Focus: A new research paper, industry study, or expert finding just landed that's relevant to this merchant's patient/customer cohort.
Message goal: Share the specific finding with source citation, connect it to the merchant's actual patient profile, offer to do work (draft content, pull abstract).
Must include: trial/study size (n=X), percentage finding, source citation (journal, page, date).
CTA: open_ended — "Want me to pull it + draft a patient-ed WhatsApp you can share?"
""",

    "regulation_change": """TRIGGER TYPE: Regulatory/Compliance Change
Focus: A regulator (DCI, FDA, GST Council, CDSCO, etc.) issued a rule change with a deadline.
Message goal: Alert the merchant urgently, explain what changes, what they need to audit, the deadline.
Must include: regulatory body name, effective date/deadline, specific change (what drops, what passes, what doesn't).
CTA: binary_yes_stop — "Want me to draft the audit checklist?"
""",

    "recall_due": """TRIGGER TYPE: Patient/Customer Recall Reminder (CUSTOMER-FACING)
Focus: A customer's recall window is open (e.g., 6-month dental cleaning, 3-month follow-up).
Message goal: Warm, specific reminder with real appointment slots and real pricing from the merchant's catalog.
Must include: customer name, time since last visit, specific available slots (date+time), real offer price.
send_as: MUST be "merchant_on_behalf"
CTA: multi_choice_slot — "Reply 1 for [slot1], 2 for [slot2]"
""",

    "perf_dip": """TRIGGER TYPE: Performance Dip Alert
Focus: A specific metric (calls, views, CTR, directions) dropped significantly in the last 7 days.
Message goal: Surface the exact number, compare to baseline/peer median, propose one specific action.
Must include: metric name, exact delta %, comparison to baseline or peer, one specific fix.
CTA: binary_yes_stop — "Want me to [specific action]?"
""",

    "perf_spike": """TRIGGER TYPE: Performance Spike
Focus: A metric spiked positively — celebrate it and suggest capitalizing on the momentum.
Message goal: Acknowledge the win with exact numbers, identify likely driver, propose capitalizing on it.
Must include: metric, delta %, likely driver if known, specific next action to sustain it.
CTA: open_ended or binary_yes_stop.
""",

    "seasonal_perf_dip": """TRIGGER TYPE: Seasonal Performance Dip (Expected — Not a Crisis)
Focus: The dip is NORMAL and seasonal. Don't alarm the merchant. Reframe it as expected and redirect energy.
Message goal: Pre-empt merchant anxiety, give industry-wide context (the benchmark range), pivot to retention.
Must include: exact dip %, the seasonal context (e.g., "every metro gym sees -25 to -35% Apr-Jun"), what to do instead.
CTA: binary_yes_stop — "Want me to draft a [retention tactic] for your X members?"
""",

    "milestone_reached": """TRIGGER TYPE: Milestone Achieved
Focus: The merchant hit a meaningful threshold (100 reviews, 500 customers, etc.).
Message goal: Celebrate genuinely, then immediately propose the next milestone or a follow-on action.
Must include: exact current value, milestone value, specific next step.
CTA: open_ended or binary_yes_stop.
""",

    "active_planning_intent": """TRIGGER TYPE: Merchant Expressed Active Planning Intent
Focus: The merchant said something in conversation that signals they want to build/create/plan something.
Message goal: DELIVER the artifact they asked for immediately — do NOT ask more qualifying questions.
Must include: a concrete draft, pricing/structure, actionable first step.
This is ACTION MODE — no more qualification. Give them what they asked for.
CTA: binary_confirm_cancel — "Reply CONFIRM to proceed, CANCEL to adjust."
""",

    "festival_upcoming": """TRIGGER TYPE: Upcoming Festival/Holiday
Focus: A festival is coming up in the next few days/weeks that's relevant to this merchant's category.
Message goal: Connect the festival to a specific, category-relevant offer or content opportunity.
Must include: festival name, days until, category-specific angle (not generic "happy festival").
CTA: binary_yes_stop — "Want me to draft the [festival] post for your GBP?"
""",

    "ipl_match_today": """TRIGGER TYPE: IPL Match Today
Focus: There's an IPL match today — but the key insight is WHICH DAY (weeknight vs weekend) matters.
Message goal: Give the CORRECT advice based on the data: weeknight matches = push promos; Saturday matches = skip promos (covers drop 12%).
Must include: match details, time, weeknight/weekend classification, the -12% or +18% data point.
CTA: binary_yes_stop — "Want me to draft the [promo/banner]?"
""",

    "supply_alert": """TRIGGER TYPE: Supply Chain Alert / Product Recall
Focus: A product batch has a quality/safety issue requiring merchant action and customer communication.
Message goal: Alert urgently with exact batch numbers, explain the risk (bounded), give the count of affected customers.
Must include: molecule/product name, specific batch numbers, manufacturer, affected customer count (derived from their chronic_rx data).
CTA: binary_yes_stop — "Want me to draft their WhatsApp note + the replacement workflow?"
Urgency: HIGH — put the key info in the first sentence.
""",

    "chronic_refill_due": """TRIGGER TYPE: Chronic Prescription Refill Due (CUSTOMER-FACING)
Focus: A customer's chronic medications are about to run out. This is sent from the pharmacy (merchant) to the customer.
Message goal: Friendly, precise refill reminder with full molecule names, total cost with discount applied, delivery details.
Must include: customer name, full molecule names (not generic "your medicines"), exact date of stock runout, total ₹ with senior/member discount applied, delivery option.
send_as: MUST be "merchant_on_behalf"
CTA: binary_confirm_cancel — "Reply CONFIRM to dispatch."
""",

    "customer_lapsed_hard": """TRIGGER TYPE: Lapsed Customer Winback (CUSTOMER-FACING)
Focus: A customer hasn't visited in 57+ days. Sent from merchant to customer.
Message goal: No-shame, warm re-engagement with a specific new offering that matches their previous stated goal/focus.
Must include: customer name, time since last visit (casual, not accusatory), a NEW specific offering that matches their previous goal, no-commitment CTA.
send_as: MUST be "merchant_on_behalf"
CTA: binary_yes_stop — "Reply YES — no commitment, no auto-charge."
""",

    "wedding_package_followup": """TRIGGER TYPE: Bridal/Wedding Package Follow-Up (CUSTOMER-FACING)
Focus: A customer did a bridal trial and their wedding is coming up. Follow up on the next preparation step.
Message goal: Count down to the wedding specifically, connect to the preparation window that's now open, offer a concrete next booking.
Must include: customer name, exact days to wedding, current preparation window/step, specific slot offer.
send_as: MUST be "merchant_on_behalf"
CTA: binary_yes_stop — "Want me to block your preferred slot?"
""",

    "curious_ask_due": """TRIGGER TYPE: Weekly Curious-Ask (Merchant Engagement Cadence)
Focus: A low-stakes, open-ended question to keep the merchant engaged and gather fresh intel.
Message goal: Ask a specific, useful question — then immediately offer to turn their answer into value (a GBP post, a WhatsApp template, a data insight).
Must include: the specific question (what's most-asked service, which slot is fullest, etc.), the specific deliverable Vera will produce from the answer.
CTA: open_ended — The question IS the CTA.
Effort externalization is the key lever here.
""",

    "winback_eligible": """TRIGGER TYPE: Lapsed Merchant Winback (Subscription Expired)
Focus: The merchant's subscription expired N days ago and performance is declining.
Message goal: Acknowledge the gap without being pushy, surface the specific performance decline since expiry, offer a concrete re-engagement path.
Must include: days since expiry, exact performance metric decline, number of new customer opportunities missed, renewal path.
CTA: binary_yes_stop — "Want to reconnect? Reply YES."
""",

    "competitor_opened": """TRIGGER TYPE: New Competitor Opened Nearby
Focus: A new business in the same category opened within X km.
Message goal: Voyeur curiosity (the merchant wants to know), then position it as an opportunity to differentiate.
Must include: competitor name, distance, their offer/price vs merchant's offer/price, one specific differentiator to push.
CTA: open_ended or binary_yes_stop.
""",

    "cde_opportunity": """TRIGGER TYPE: Continuing Education / Professional Development Opportunity
Focus: A webinar, training, or certification event is upcoming and relevant to this merchant.
Message goal: Brief, specific event pitch with credit/cost details.
Must include: event name, date, credits, cost, speaker/topic specifics.
CTA: binary_yes_stop — "Interested? Reply YES and I'll send the registration link."
""",

    "review_theme_emerged": """TRIGGER TYPE: Review Pattern Emerged
Focus: Multiple recent reviews mention the same theme (positive or negative). This is an actionable insight.
Message goal: Surface the pattern with exact count and a verbatim quote, then propose a specific operational or marketing action.
Must include: theme name, occurrence count (last 30d), a verbatim (or near-verbatim) quote from the reviews, trend direction.
CTA: binary_yes_stop — "Want me to draft a [response template / GBP post / operational fix]?"
""",

    "dormant_with_vera": """TRIGGER TYPE: Merchant Dormant (No Reply for 14+ Days)
Focus: The merchant hasn't engaged with Vera in a while. This is a low-key re-engagement.
Message goal: Lightweight check-in — NOT a sales pitch. Ask a simple question or share one piece of value.
Must NOT: start with "I noticed you haven't replied" or any guilt-trip framing.
CTA: open_ended — keep it casual.
""",

    "gbp_unverified": """TRIGGER TYPE: Google Business Profile Unverified
Focus: The merchant's GBP is unverified, which suppresses their local search visibility.
Message goal: Surface the specific uplift they're missing (the estimated %) and make the verification path sound easy.
Must include: estimated uplift %, the exact verification path (postcard or phone), what gets unlocked.
CTA: binary_yes_stop — "Want me to walk you through the verification? 5 minutes."
""",

    "trial_followup": """TRIGGER TYPE: Trial Class/Service Follow-Up (CUSTOMER-FACING)
Focus: A customer attended a trial class/service. Follow up to convert them to a regular membership/booking.
Message goal: Warm, specific follow-up referencing the trial experience, with a concrete next session slot.
Must include: customer name, trial date, specific next session date+time, trial-to-membership offer.
send_as: MUST be "merchant_on_behalf"
CTA: binary_yes_stop — "Want me to hold this spot for you? Reply YES."
""",

    "category_seasonal": """TRIGGER TYPE: Category-Wide Seasonal Demand Shift
Focus: Seasonal demand patterns are shifting (e.g., summer → ORS up, cold medicine down).
Message goal: Give the merchant specific data on what's trending up/down, and what shelf/operational action to take.
Must include: specific products/categories with % changes, the seasonal pattern, concrete shelf/stock action.
CTA: binary_yes_stop — "Want me to draft a shelf-rearrange plan?"
""",

    "renewal_due": """TRIGGER TYPE: Subscription Renewal Due
Focus: The merchant's magicpin subscription is expiring in N days.
Message goal: Surface the specific value they got during the subscription period, then make renewing feel low-friction.
Primary signals: days_remaining from subscription context + views/calls delivered in the period.
Must include: days remaining, plan name, specific value delivered (views, calls, leads in the period), renewal amount.
CTA: binary_yes_stop — "Reply YES to renew, or I can show you the upgrade options."
""",

    "engaged_reply": """TRIGGER TYPE: Engaged Merchant Reply (Conversation Continuation)
Focus: The merchant replied with a question, interest signal, or partial engagement. Advance — don't repeat.
CRITICAL: Match the category voice exactly (see CATEGORY CONTEXT → Voice tone above).
Message goal: Move the conversation forward toward one specific action. Do NOT re-qualify.
  - If merchant asked a question → answer it precisely using data from contexts.
  - If merchant showed interest → deliver the next concrete step or artifact.
  - If merchant is exploring → narrow to ONE option with a real number attached.
Must NOT: Ask another qualifying question if Vera already asked one. Don't repeat the opening hook.
CTA: binary_confirm_cancel if asking for commitment, open_ended if asking for their input.
""",

    "customer_lapsed_soft": """TRIGGER TYPE: Soft Lapsed Customer Re-engagement (CUSTOMER-FACING)
Focus: Customer has been inactive for 30-56 days — early enough for a warm re-engagement.
Message goal: Warm, casual re-engagement that references their specific last service and offers something new.
Primary signals: last_visit date, services_received, any new offers matching their preferences.
Must include: customer name, time since last visit, their specific last service, one new/relevant offer.
send_as: MUST be "merchant_on_behalf"
CTA: binary_yes_stop — "Want to book? Reply YES."
""",

    "appointment_tomorrow": """TRIGGER TYPE: Appointment Reminder (CUSTOMER-FACING)
Focus: Customer has an appointment tomorrow. Friendly reminder to confirm.
Message goal: Confirm appointment details and reduce no-shows.
Primary signals: customer name, appointment date/time from trigger payload, merchant name.
Must include: customer name, appointment date, time, service name, and a confirm/reschedule CTA.
send_as: MUST be "merchant_on_behalf"
CTA: binary_confirm_cancel — "Reply CONFIRM to keep it, CANCEL to reschedule."
""",
}

# Default framing for unknown trigger kinds — still specificity-enforced
DEFAULT_FRAMING = """TRIGGER TYPE: General Merchant Outreach
Focus: Send a relevant, useful, context-grounded message based on the trigger payload and merchant state.
Primary signal: Check trigger.payload first — what specific event/data triggered this?
Then: use the ⚡ PRE-COMPUTED DEMAND SIGNALS to add a real number.
Must include: at least 1 number from merchant performance + 1 from trigger payload or offers.
CTA: binary_yes_stop or open_ended based on whether you're asking for a commit or sharing info.
"""


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

async def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """
    Compose a WhatsApp message from the 4 context blocks.
    Returns a dict with: body, cta, send_as, suppression_key, rationale,
    template_name, template_params.
    """
    trigger_kind = trigger.get("kind", "unknown")
    framing = TRIGGER_FRAMING.get(trigger_kind, DEFAULT_FRAMING)

    # Build the user prompt (all 4 contexts injected)
    user_prompt = _build_user_prompt(category, merchant, trigger, customer, conversation_history or [])

    full_system = MASTER_SYSTEM + "\n\n## THIS MESSAGE'S SPECIFIC FRAMING\n" + framing

    try:
        raw = await call_gemini(full_system, user_prompt)
        result = extract_json(raw)
        result = _validate_and_fix(result, trigger, category, merchant, customer, conversation_history or [])
        return result
    except Exception as e:
        # Fallback: safe minimal message grounded in merchant name + trigger kind
        return _safe_fallback(trigger, merchant, customer, str(e))


# ---------------------------------------------------------------------------
# User prompt builder — injects all 4 contexts as structured data
# ---------------------------------------------------------------------------

def _build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    conversation_history: list,
) -> str:
    mid = merchant.get("identity", {}).get("name", "Merchant")
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    city = merchant.get("identity", {}).get("city", "")
    locality = merchant.get("identity", {}).get("locality", "")
    languages = merchant.get("identity", {}).get("languages", ["en"])
    lang_str = ", ".join(languages)

    perf = merchant.get("performance", {})
    signals = merchant.get("signals", [])
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg = merchant.get("customer_aggregate", {})
    conv_hist = merchant.get("conversation_history", []) or conversation_history

    # Category digest — most relevant items
    digest = category.get("digest", [])
    peer = category.get("peer_stats", {})
    voice = category.get("voice", {})
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])

    # Trigger payload
    t_payload = trigger.get("payload", {})
    t_kind = trigger.get("kind", "")
    t_scope = trigger.get("scope", "merchant")
    t_urgency = trigger.get("urgency", 2)
    t_suppression = trigger.get("suppression_key", "")
    t_expires = trigger.get("expires_at", "")

    # Resolve digest item referenced in trigger (if any)
    top_item = None
    top_item_id = t_payload.get("top_item_id") or t_payload.get("alert_id") or t_payload.get("digest_item_id")
    if top_item_id:
        for d in digest:
            if d.get("id") == top_item_id:
                top_item = d
                break

    # Customer context
    cust_block = ""
    if customer:
        cid = customer.get("identity", {})
        rel = customer.get("relationship", {})
        prefs = customer.get("preferences", {})
        cst = customer.get("state", "unknown")
        cust_block = f"""
## CUSTOMER CONTEXT (message is ON BEHALF of merchant, sent TO this customer)
Name: {cid.get("name", "Customer")}
Language pref: {cid.get("language_pref", "english")}
Age band: {cid.get("age_band", "unknown")}
State: {cst}
First visit: {rel.get("first_visit", "unknown")}
Last visit: {rel.get("last_visit", "unknown")}
Visits total: {rel.get("visits_total", 0)}
Services received: {rel.get("services_received", [])}
Lifetime value: ₹{rel.get("lifetime_value", 0)}
Preferred slots: {prefs.get("preferred_slots", "unknown")}
Channel: {prefs.get("channel", "whatsapp")}
Consent scope: {customer.get("consent", {}).get("scope", [])}
Extra prefs: {json.dumps({k: v for k, v in prefs.items() if k not in ("preferred_slots", "channel")})}
"""

    # Conversation history summary
    hist_lines = []
    for h in (conv_hist or [])[-6:]:  # last 6 turns max
        role = h.get("from", "?")
        body_preview = (h.get("body", "") or "")[:120]
        hist_lines.append(f"  [{role}] {body_preview}")
    hist_block = "\n".join(hist_lines) if hist_lines else "  (no prior conversation)"

    prompt = f"""## CATEGORY CONTEXT
Slug: {category.get("slug", "unknown")}
Voice tone: {voice.get("tone", "unknown")}
Vocab allowed: {voice.get("vocab_allowed", [])[:8]}
Vocab TABOO (never use): {voice.get("vocab_taboo", [])}
Salutation examples: {voice.get("salutation_examples", [])}

Peer stats (benchmark this merchant against):
  avg_rating={peer.get("avg_rating")}, avg_ctr={peer.get("avg_ctr")}, avg_views_30d={peer.get("avg_views_30d")}, avg_calls_30d={peer.get("avg_calls_30d")}, avg_review_count={peer.get("avg_review_count")}

Digest items (this week's knowledge — cite these, don't invent):
{json.dumps(digest, indent=2, ensure_ascii=False)}

Seasonal beats: {json.dumps(seasonal, ensure_ascii=False)}
Trend signals: {json.dumps(trends, ensure_ascii=False)}

---

## MERCHANT CONTEXT
Name: {mid}
Owner first name: {owner}
City: {city}, Locality: {locality}
Languages: {lang_str}  ← MUST match this in your message
Verified: {merchant.get("identity", {}).get("verified", False)}
Subscription: status={merchant.get("subscription", {}).get("status")}, plan={merchant.get("subscription", {}).get("plan")}, days_remaining={merchant.get("subscription", {}).get("days_remaining")}

Performance (30d): views={perf.get("views")}, calls={perf.get("calls")}, directions={perf.get("directions")}, ctr={perf.get("ctr")}, leads={perf.get("leads")}
7d delta: {json.dumps(perf.get("delta_7d", {}))}
vs peer median CTR: {perf.get("ctr", 0)} vs peer {peer.get("avg_ctr", 0)} → {"BELOW" if perf.get("ctr", 0) < peer.get("avg_ctr", 0) else "ABOVE"} peer median

⚡ PRE-COMPUTED DEMAND SIGNALS — pick 2+ and use them verbatim in your message:
  · DEMAND: {perf.get("views", 0):,} people viewed this listing in 30 days (~{round(perf.get("views", 0) / 30)} per day)
  · GAP: only {perf.get("calls", 0)} called from {perf.get("views", 0):,} views → {round((1 - perf.get("calls", 0) / max(perf.get("views", 0), 1)) * 100, 1)}% of viewers don't convert
  · PEER GAP: CTR is {"BELOW" if perf.get("ctr", 0) < peer.get("avg_ctr", 0) else "ABOVE"} peer median by {abs(round((perf.get("ctr", 0) - peer.get("avg_ctr", 0)) / max(peer.get("avg_ctr", 0.001), 0.001) * 100, 1))}%
  · LOCAL TREND: "{trends[0].get("query", "N/A") if trends else "N/A"}" searches up {round((trends[0].get("delta_yoy", 0) if trends else 0) * 100, 0):.0f}% YoY in this locality
  · LAPSED: {cust_agg.get("lapsed_180d_plus", 0)} lapsed customers (180d+) out of {cust_agg.get("total_unique_ytd", 0)} total — winback opportunity

Active offers: {json.dumps(active_offers, ensure_ascii=False)}
All offers: {json.dumps(merchant.get("offers", []), ensure_ascii=False)}

Customer aggregate: {json.dumps(cust_agg, ensure_ascii=False)}
Signals: {signals}
Review themes: {json.dumps(merchant.get("review_themes", []), ensure_ascii=False)}

---

## TRIGGER CONTEXT
ID: {trigger.get("id")}
Kind: {t_kind}
Scope: {t_scope}  ← if "customer", send_as MUST be "merchant_on_behalf"
Source: {trigger.get("source")}
Urgency: {t_urgency}/5
Suppression key (copy EXACTLY): {t_suppression}
Expires: {t_expires}
Payload: {json.dumps(t_payload, indent=2, ensure_ascii=False)}

{f"RESOLVED DIGEST ITEM (the trigger references this — cite it):\n{json.dumps(top_item, indent=2, ensure_ascii=False)}" if top_item else ""}

---
{cust_block}
## PRIOR CONVERSATION (do NOT repeat what was already said)
{hist_block}

---

## YOUR TASK
Compose a single WhatsApp message. Output ONLY the JSON object.

Checklist before you output:
☑ Does the body contain at least 2 specific numbers from the contexts? (mandatory)
☑ Does it use the correct category voice and salutation style?
☑ Does it reference at least 1 active offer by name/price?
☑ Is the CTA the final sentence?
☑ Have I cited 2+ compulsion levers in the rationale?
☑ Is the suppression_key exactly: {t_suppression}

If any checkbox fails, rewrite the body before outputting.
"""
    return prompt


# ---------------------------------------------------------------------------
# Post-LLM validation and fix-up
# ---------------------------------------------------------------------------

def _validate_and_fix(
    result: dict,
    trigger: dict,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
    conversation_history: list,
) -> dict:
    """Apply hard rules to the composed message. Fix what's fixable; flag the rest."""

    voice = category.get("voice", {})
    taboos = [t.lower() for t in voice.get("vocab_taboo", [])]

    # 1. Ensure suppression_key matches trigger
    t_suppression = trigger.get("suppression_key", "")
    if t_suppression:
        result["suppression_key"] = t_suppression

    # 2. Enforce send_as based on trigger scope
    if trigger.get("scope") == "customer":
        result["send_as"] = "merchant_on_behalf"
    else:
        result.setdefault("send_as", "vera")

    # 3. Remove any URLs from body (Meta rejects them)
    body = result.get("body", "")
    body = re.sub(r"https?://\S+", "[link removed]", body)
    result["body"] = body.strip()

    # 4. Check for taboo words
    body_lower = body.lower()
    for taboo in taboos:
        if taboo in body_lower and len(taboo) > 4:
            result["rationale"] = result.get("rationale", "") + f" [WARNING: possible taboo '{taboo}' detected — review manually]"

    # 5. Ensure CTA is valid
    valid_ctas = {"binary_yes_stop", "binary_yes_no", "binary_confirm_cancel", "open_ended", "multi_choice_slot", "none"}
    if result.get("cta") not in valid_ctas:
        if trigger.get("scope") == "customer":
            result["cta"] = "binary_yes_stop"
        else:
            result["cta"] = "open_ended"

    # 6. Ensure template_name and template_params exist
    if not result.get("template_name"):
        trigger_kind = trigger.get("kind", "general")
        result["template_name"] = f"vera_{trigger_kind}_v1"
    if not result.get("template_params"):
        owner = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "")
        result["template_params"] = [owner, trigger.get("kind", ""), ""]

    # 7. Ensure rationale exists
    if not result.get("rationale"):
        result["rationale"] = f"Composed for {trigger.get('kind')} trigger targeting {merchant.get('identity', {}).get('name', 'merchant')}."

    # 8. Specificity check — warn if body has no numbers (low specificity risk)
    numbers_found = re.findall(r"[₹\d][\d,\.]*(?:%|₹|\d)", body)
    digit_words = re.findall(r"\b\d+\b", body)
    if not numbers_found and not digit_words:
        result["rationale"] = result.get("rationale", "") + " [LOW SPECIFICITY: no numbers in body — consider re-prompting]"

    return result


# ---------------------------------------------------------------------------
# Safe fallback — used when LLM fails entirely
# ---------------------------------------------------------------------------

def _safe_fallback(trigger: dict, merchant: dict, customer: Optional[dict], error: str) -> dict:
    """Minimal safe message when LLM composition fails. Uses merchant data for specificity."""
    owner = merchant.get("identity", {}).get("owner_first_name", "") if merchant else ""
    name = merchant.get("identity", {}).get("name", "there") if merchant else "there"
    locality = merchant.get("identity", {}).get("locality", "") if merchant else ""
    t_kind = trigger.get("kind", "update")
    t_scope = trigger.get("scope", "merchant")
    t_suppression = trigger.get("suppression_key", f"fallback:{t_kind}")

    greeting = owner or name

    if customer:
        cust_name = customer.get("identity", {}).get("name", "")
        cust_merchant = merchant.get("identity", {}).get("name", "") if merchant else ""
        last_visit = customer.get("relationship", {}).get("last_visit", "")
        last_visit_str = f" (last visit: {last_visit})" if last_visit else ""
        body = f"Hi {cust_name}, {cust_merchant} here{last_visit_str}. We have an update for you — reply YES if you'd like to know more, STOP to opt out."
        send_as = "merchant_on_behalf"
    else:
        # Build a specific fallback using merchant performance data
        perf = merchant.get("performance", {}) if merchant else {}
        views = perf.get("views", 0)
        calls = perf.get("calls", 0)
        active_offers = [o for o in (merchant.get("offers", []) if merchant else []) if o.get("status") == "active"]
        best_offer = active_offers[0].get("title", "") if active_offers else ""
        loc_str = f"in {locality}" if locality else "on magicpin"

        if views and best_offer:
            body = (
                f"Hi {greeting}! {views:,} people viewed your listing {loc_str} this month"
                f" — but only {calls} called. Your {best_offer} is live."
                f" Should I draft a WhatsApp to close that gap? Reply YES."
            )
        elif views:
            body = (
                f"Hi {greeting}! {views:,} people searched for you {loc_str} this month."
                f" There's something worth acting on — reply YES to hear more, STOP to opt out."
            )
        else:
            body = (
                f"Hi {greeting}! Quick update from Vera — there's a relevant signal"
                f" for {name} right now. Reply YES to hear more, STOP to opt out."
            )
        send_as = "vera"

    return {
        "body": body,
        "cta": "binary_yes_stop",
        "send_as": send_as,
        "suppression_key": t_suppression,
        "rationale": f"Grounded fallback: LLM unavailable ({error[:80]}). Used merchant views/offers from context.",
        "template_name": f"vera_{t_kind}_fallback_v1",
        "template_params": [greeting or name, t_kind, locality or ""],
    }
