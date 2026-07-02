# Vera — magicpin Merchant AI Bot

**Candidate**: Ansh Sharma | ansh.sharma@kalvium.community

---

## Approach

### Architecture

A stateful FastAPI HTTP service with 5 endpoints, backed by an in-memory context store and an LLM-powered composer.

```
Judge → POST /v1/context  → Context Store (versioned, atomic replace)
Judge → POST /v1/tick     → Composer Engine → LLM (Gemini 2.0 Flash) → Actions[]
Judge → POST /v1/reply    → Conversation Handler → LLM → {send|wait|end}
```

### The Composer Engine (`composer.py`)

**Trigger-kind dispatch**: Each of 20+ trigger kinds gets a specialized framing prompt that sets the "why now" anchor before the LLM runs. A `research_digest` trigger gets a citation-focused framing; a `recall_due` trigger gets a slot-booking framing. The master system prompt (shared across all kinds) encodes:

- All 5 category voice profiles (dentist clinical, salon warm-practical, etc.)
- All 8 compulsion levers (specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, asking-the-merchant, single binary CTA)
- Hard constraints: no URLs, no fabrication, no taboo words, single CTA
- Language preference matching rules

**Post-LLM validation**: Every output is checked for taboo words, correct `send_as` (customer trigger → `merchant_on_behalf`), valid CTA shape, suppression key integrity, and body non-repetition before returning.

### Context Grounding (the key differentiator)

The context store holds all 4 context types with **version-aware atomic replace**. When the judge injects a new digest item mid-test (e.g., a new DCI compliance note), the store replaces the old category context immediately. The next composition automatically uses the fresh data — no extra code needed.

This is why bots that pattern-match the simulator fail on fresh scenarios: they bake in old context. Our bot always reads from the latest store state.

### Multi-Turn Conversation (`conversation.py`)

Three sequential detectors on every `/v1/reply`:

1. **Auto-Reply Detector**: Regex patterns for WA Business canned responses. Turn 1 → bridging message. Turn 2 → `wait` 4h. Turn 3+ → `end`.
2. **Intent Detector**: Classifies as action/reject/hostile/out-of-scope/engaged. On "let's do it" → switches to ACTION_MODE (delivers the artifact, no more qualifying questions).
3. **Graceful Exit**: Hostile/explicit rejection → polite `end`. Out-of-scope → decline + redirect back to thread.

### LLM Choice

**Gemini 2.0 Flash** at `temperature=0` for deterministic, fast output. The Gemini JSON mode (`responseMimeType: "application/json"`) reduces parse failures. Average latency ~3-5s per composition, well within the 30s judge deadline.

### What I Would Do With More Context

1. **Real merchant customer CRM data**: The customer aggregate fields are synthesized. Real refill schedules, actual appointment slot availability, and real visit history would make customer-facing messages dramatically more precise.
2. **Merchant reply history at scale**: With 6,000-10,000 daily merchant conversations, pattern-matching on which trigger kinds generate the highest reply rates would let me A/B the prompt variants objectively.
3. **Per-city peer benchmarks**: Currently using category-level peer stats. City-locality level benchmarks (e.g., "Lajpat Nagar dentists specifically") would make social-proof levers more credible.

---

## Deployment

Railway — auto-deploys from GitHub. Set `GEMINI_API_KEY` in Railway environment variables.

```bash
# Local testing
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
uvicorn bot:app --host 0.0.0.0 --port 8080

# Run judge simulator
python judge_simulator.py

# Generate submission.jsonl
python generate_submission.py --dataset-dir ./dataset
```

---

## File Structure

```
bot.py                  — FastAPI app (5 endpoints)
composer.py             — LLM composer (trigger-kind dispatch + prompts)
conversation.py         — Multi-turn handler (auto-reply, intent, exit)
store.py                — In-memory context store (versioned, indexed)
llm_client.py           — Async Gemini REST client
config.py               — Environment config
generate_submission.py  — Generates submission.jsonl from 30 test pairs
requirements.txt
Procfile / railway.json — Railway deployment
```
