"""
generate_submission.py — Run the bot against the 30 canonical test pairs
and produce submission.jsonl.

Usage:
    python generate_submission.py [--dataset-dir ./dataset/expanded] [--out submission.jsonl]

The bot must NOT be running when you run this — this script calls composer.py
directly, not the HTTP server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def load_dataset(dataset_dir: Path) -> tuple[dict, dict, dict, dict]:
    """Load all 4 context types from the expanded dataset directory."""
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    # Categories
    cat_dir = dataset_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            categories[data["slug"]] = data
    else:
        # Fallback to seed categories
        seed_dir = dataset_dir.parent / "categories"
        for f in seed_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            categories[data["slug"]] = data

    # Merchants
    merch_dir = dataset_dir / "merchants"
    if merch_dir.exists():
        for f in merch_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            mid = data.get("merchant_id", f.stem)
            merchants[mid] = data
    else:
        seed = dataset_dir.parent / "merchants_seed.json"
        if seed.exists():
            raw = json.loads(seed.read_text(encoding="utf-8"))
            for m in raw.get("merchants", []):
                merchants[m["merchant_id"]] = m

    # Customers
    cust_dir = dataset_dir / "customers"
    if cust_dir.exists():
        for f in cust_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            cid = data.get("customer_id", f.stem)
            customers[cid] = data
    else:
        seed = dataset_dir.parent / "customers_seed.json"
        if seed.exists():
            raw = json.loads(seed.read_text(encoding="utf-8"))
            for c in raw.get("customers", []):
                customers[c["customer_id"]] = c

    # Triggers
    trg_dir = dataset_dir / "triggers"
    if trg_dir.exists():
        for f in trg_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            tid = data.get("id", f.stem)
            triggers[tid] = data
    else:
        seed = dataset_dir.parent / "triggers_seed.json"
        if seed.exists():
            raw = json.loads(seed.read_text(encoding="utf-8"))
            for t in raw.get("triggers", []):
                triggers[t["id"]] = t

    return categories, merchants, customers, triggers


def load_test_pairs(dataset_dir: Path) -> list[dict]:
    """Load the 30 canonical test pairs."""
    # Try expanded dataset first
    pairs_file = dataset_dir / "test_pairs.json"
    if pairs_file.exists():
        return json.loads(pairs_file.read_text(encoding="utf-8"))

    # Fallback: generate test pairs from seed data (first 25 triggers)
    print("[WARN] test_pairs.json not found — using first 25 seed triggers as test pairs.")
    seed = dataset_dir.parent / "triggers_seed.json"
    if not seed.exists():
        raise FileNotFoundError(f"No test_pairs.json and no triggers_seed.json in {dataset_dir.parent}")

    raw = json.loads(seed.read_text(encoding="utf-8"))
    pairs = []
    for i, trg in enumerate(raw.get("triggers", [])[:25], 1):
        pairs.append({
            "test_id": f"T{i:02d}",
            "trigger_id": trg["id"],
            "merchant_id": trg.get("merchant_id", ""),
            "customer_id": trg.get("customer_id"),
        })
    return pairs


async def run(dataset_dir: Path, out_file: Path):
    from composer import compose

    print(f"Loading dataset from {dataset_dir}...")
    categories, merchants, customers, triggers = load_dataset(dataset_dir)
    print(f"  Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    test_pairs = load_test_pairs(dataset_dir)
    print(f"  Test pairs: {len(test_pairs)}")

    lines = []
    for pair in test_pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair.get("merchant_id", "")
        customer_id = pair.get("customer_id")

        trigger = triggers.get(trigger_id)
        if not trigger:
            print(f"  [SKIP] {test_id}: trigger {trigger_id} not found")
            continue

        # Resolve merchant_id from trigger if not in pair
        if not merchant_id:
            merchant_id = trigger.get("merchant_id", "")

        merchant = merchants.get(merchant_id)
        if not merchant:
            print(f"  [SKIP] {test_id}: merchant {merchant_id} not found")
            continue

        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug)
        if not category:
            print(f"  [SKIP] {test_id}: category {cat_slug} not found")
            continue

        customer = customers.get(customer_id) if customer_id else None

        print(f"  Composing {test_id}: {trigger.get('kind')} → {merchant.get('identity', {}).get('name', merchant_id)}")

        try:
            result = await compose(category, merchant, trigger, customer)
            line = {
                "test_id": test_id,
                "trigger_id": trigger_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", trigger.get("suppression_key", "")),
                "rationale": result.get("rationale", ""),
                "template_name": result.get("template_name", ""),
                "template_params": result.get("template_params", []),
            }
            lines.append(line)
            print(f"    ✓ {len(result.get('body', ''))} chars, cta={result.get('cta')}, send_as={result.get('send_as')}")
        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            # Write a placeholder so we still have 30 lines
            lines.append({
                "test_id": test_id,
                "trigger_id": trigger_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "body": f"[ERROR composing {test_id}: {str(e)[:100]}]",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Composition failed: {str(e)[:200]}",
                "template_name": "",
                "template_params": [],
            })

    # Write JSONL
    out_file.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
        encoding="utf-8",
    )
    print(f"\nWrote {len(lines)} lines to {out_file}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate submission.jsonl from the 30 test pairs.")
    parser.add_argument("--dataset-dir", default="./dataset/expanded", help="Path to expanded dataset directory")
    parser.add_argument("--out", default="submission.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_file = Path(args.out)

    asyncio.run(run(dataset_dir, out_file))


if __name__ == "__main__":
    main()
