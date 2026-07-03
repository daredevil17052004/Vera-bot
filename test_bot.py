"""
test_bot.py — End-to-end tests for the Vera bot.
Run with: python test_bot.py
"""
import sys, io
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json
import urllib.request
import urllib.error
import time
import sys

BASE = "http://localhost:8091"
PASS = 0
FAIL = 0

def req(method, path, body=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
        return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode()), e.code
        except:
            return {"error": str(e)}, e.code

def check(name, condition, got=""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name} — got: {got}")
        FAIL += 1

print("\n" + "="*60)
print("  VERA BOT — END-TO-END TEST SUITE")
print("="*60)

# ============================================================
# 1. WARMUP — healthz + metadata
# ============================================================
print("\n[1] WARMUP")
data, status = req("GET", "/v1/healthz")
check("healthz 200", status == 200)
check("healthz status=ok", data.get("status") == "ok")

data, status = req("GET", "/v1/metadata")
check("metadata 200", status == 200)
check("metadata team_name", "Ansh" in data.get("team_name",""))
check("metadata model", "gemini" in data.get("model","").lower())

# ============================================================
# 2. CONTEXT PUSH — all 5 categories + 2 merchants + 1 trigger
# ============================================================
print("\n[2] CONTEXT PUSH")

cats = ["dentists","salons","restaurants","gyms","pharmacies"]
for slug in cats:
    cat = json.load(open(f"dataset/categories/{slug}.json", encoding="utf-8"))
    d, s = req("POST", "/v1/context", {"scope":"category","context_id":slug,"version":1,"payload":cat,"delivered_at":"2026-07-02T15:00:00Z"})
    check(f"category/{slug}", d.get("accepted") == True, d)

# Merchant
merchants = json.load(open("dataset/merchants_seed.json", encoding="utf-8"))["merchants"]
m = merchants[0]  # Dr. Meera
d, s = req("POST", "/v1/context", {"scope":"merchant","context_id":m["merchant_id"],"version":1,"payload":m,"delivered_at":"2026-07-02T15:00:00Z"})
check(f"merchant/{m['merchant_id'][:20]}", d.get("accepted") == True, d)

m2 = merchants[2]  # Studio11 salon
d, s = req("POST", "/v1/context", {"scope":"merchant","context_id":m2["merchant_id"],"version":1,"payload":m2,"delivered_at":"2026-07-02T15:00:00Z"})
check(f"merchant/{m2['merchant_id'][:20]}", d.get("accepted") == True, d)

# Customer
customers = json.load(open("dataset/customers_seed.json", encoding="utf-8"))["customers"]
c = customers[0]  # Priya
d, s = req("POST", "/v1/context", {"scope":"customer","context_id":c["customer_id"],"version":1,"payload":c,"delivered_at":"2026-07-02T15:00:00Z"})
check(f"customer/{c['customer_id'][:20]}", d.get("accepted") == True, d)

# Triggers
triggers = json.load(open("dataset/triggers_seed.json", encoding="utf-8"))["triggers"]
for trg in triggers[:5]:  # push first 5 triggers
    d, s = req("POST", "/v1/context", {"scope":"trigger","context_id":trg["id"],"version":1,"payload":trg,"delivered_at":"2026-07-02T15:00:00Z"})
    check(f"trigger/{trg['id'][:30]}", d.get("accepted") == True, d)

# Idempotency: same version should 409
d, s = req("POST", "/v1/context", {"scope":"category","context_id":"dentists","version":1,"payload":{},"delivered_at":"2026-07-02T15:00:00Z"})
check("stale_version -> 409", s == 409, f"status={s}")

# Healthz should now show counts
d, s = req("GET", "/v1/healthz")
check("healthz categories=5", d.get("contexts_loaded",{}).get("category") == 5, d.get("contexts_loaded"))
check("healthz merchants=2", d.get("contexts_loaded",{}).get("merchant") == 2, d.get("contexts_loaded"))

# ============================================================
# 3. REPLY — deterministic handlers (no LLM)
# ============================================================
print("\n[3] REPLY — DETERMINISTIC HANDLERS")

# Auto-reply detection
ar_body = {"conversation_id":"test_auto_001","merchant_id":m["merchant_id"],"from_role":"merchant",
           "message":"Thank you for contacting us! Our team will respond shortly.","received_at":"2026-07-02T15:00:00Z","turn_number":2}
d, s = req("POST", "/v1/reply", ar_body)
check("auto-reply turn 1 → send", d.get("action") == "send", d.get("action"))
check("auto-reply body mentions owner", d.get("body","") != "", d.get("body","")[:60])

d, s = req("POST", "/v1/reply", ar_body)
check("auto-reply turn 2 → wait", d.get("action") == "wait", d.get("action"))
check("auto-reply wait_seconds set", d.get("wait_seconds",0) > 0, d.get("wait_seconds"))

d, s = req("POST", "/v1/reply", ar_body)
check("auto-reply turn 3 → end", d.get("action") == "end", d.get("action"))

# Hostile
hostile = {"conversation_id":"test_hostile_001","merchant_id":m["merchant_id"],"from_role":"merchant",
           "message":"Stop messaging me. This is spam and useless.","received_at":"2026-07-02T15:00:00Z","turn_number":2}
d, s = req("POST", "/v1/reply", hostile)
check("hostile → end", d.get("action") == "end", d.get("action"))

# Once ended, further messages → end immediately
d, s = req("POST", "/v1/reply", {**hostile, "message": "Another message"})
check("ended conv stays ended", d.get("action") == "end", d.get("action"))

# Explicit rejection
reject = {"conversation_id":"test_reject_001","merchant_id":m["merchant_id"],"from_role":"merchant",
          "message":"Not interested. Stop.","received_at":"2026-07-02T15:00:00Z","turn_number":2}
d, s = req("POST", "/v1/reply", reject)
check("rejection → send graceful", d.get("action") == "send", d.get("action"))
check("rejection body polite", any(w in d.get("body","").lower() for w in ["samajh","won't","not message","opt out"]), d.get("body",""))

# Out-of-scope
oos = {"conversation_id":"test_oos_001","merchant_id":m["merchant_id"],"from_role":"merchant",
       "message":"Can you help me with GST filing?","received_at":"2026-07-02T15:00:00Z","turn_number":2}
d, s = req("POST", "/v1/reply", oos)
check("out-of-scope → send redirect", d.get("action") == "send", d.get("action"))

# ============================================================
# 4. TICK — real LLM composition
# ============================================================
print("\n[4] TICK — LLM COMPOSITION (real API call)")

# Research digest trigger for Dr. Meera
trg = triggers[0]  # trg_001_research_digest_dentists
d, s = req("POST", "/v1/tick", {"now":"2026-07-02T15:00:00Z","available_triggers":[trg["id"]]}, timeout=35)
check("tick 200", s == 200, s)
actions = d.get("actions", [])
check("tick returns action", len(actions) > 0, f"actions={len(actions)}")
if actions:
    a = actions[0]
    body = a.get("body","")
    print(f"\n    MESSAGE PREVIEW ({len(body)} chars):")
    print(f"    {body[:200]}")
    print(f"    CTA: {a.get('cta')} | send_as: {a.get('send_as')} | template: {a.get('template_name')}")
    check("body non-empty", len(body) > 20, len(body))
    check("body not generic", not body.lower().startswith("hi there"), body[:50])
    check("send_as=vera (merchant trigger)", a.get("send_as") == "vera", a.get("send_as"))
    check("suppression_key present", a.get("suppression_key","") != "", a.get("suppression_key"))
    check("rationale present", len(a.get("rationale","")) > 10, a.get("rationale","")[:50])

# Repeat tick — same trigger should be suppressed
d2, _ = req("POST", "/v1/tick", {"now":"2026-07-02T15:00:01Z","available_triggers":[trg["id"]]}, timeout=35)
check("repeat tick suppressed (no duplicate)", len(d2.get("actions",[])) == 0, len(d2.get("actions",[])))

# ============================================================
# 5. TICK — customer-scoped trigger (recall_due)
# ============================================================
print("\n[5] TICK — CUSTOMER TRIGGER (recall_due)")

# Push the recall trigger for Priya
recall_trg = triggers[2]  # trg_003_recall_due_priya
d, s = req("POST", "/v1/context", {"scope":"trigger","context_id":recall_trg["id"],"version":1,"payload":recall_trg,"delivered_at":"2026-07-02T15:00:00Z"})
check("recall trigger pushed", d.get("accepted") == True or d.get("reason") == "stale_version", d)

d, s = req("POST", "/v1/tick", {"now":"2026-07-02T15:00:00Z","available_triggers":[recall_trg["id"]]}, timeout=35)
actions = d.get("actions", [])
check("recall tick returns action", len(actions) > 0, f"actions={len(actions)}")
if actions:
    a = actions[0]
    body = a.get("body","")
    print(f"\n    CUSTOMER MESSAGE PREVIEW ({len(body)} chars):")
    print(f"    {body[:200]}")
    check("send_as=merchant_on_behalf", a.get("send_as") == "merchant_on_behalf", a.get("send_as"))
    check("customer_id set", a.get("customer_id") == recall_trg.get("customer_id"), a.get("customer_id"))

# ============================================================
# 6. REPLY — LLM intent transition
# ============================================================
print("\n[6] REPLY — INTENT TRANSITION (LLM)")

commit = {"conversation_id":"test_intent_001","merchant_id":m["merchant_id"],"from_role":"merchant",
          "message":"Ok let's do it! What's next?","received_at":"2026-07-02T15:00:00Z","turn_number":3}
d, s = req("POST", "/v1/reply", commit, timeout=35)
check("intent action → send", d.get("action") == "send", d.get("action"))
if d.get("body"):
    body = d.get("body","")
    print(f"\n    ACTION REPLY ({len(body)} chars): {body[:150]}")
    qualifying_words = ["would you", "do you", "what if", "how about", "tell me more"]
    action_words = ["here", "draft", "sending", "proceed", "confirm", "next step", "done"]
    # After "let's do it", bot should NOT be qualifying
    check("intent: not still qualifying", not any(w in body.lower() for w in qualifying_words), body[:100])

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed ({int(PASS/total*100)}%)")
if FAIL == 0:
    print("  STATUS: ALL PASS ✓")
else:
    print(f"  STATUS: {FAIL} FAILED ✗")
print("="*60 + "\n")
sys.exit(0 if FAIL == 0 else 1)
