"""
Accuracy test for the altered-facts repairs in repairs.py (RS-62). Runs offline: no server,
no LLM, no embeddings.

For every sampled question and several passage-set sizes k (2, 3, 5, 10 passages, gold
passages first) it runs two cases:

  clean    the real passages. Any corrected/dropped passage here is a false positive.
  altered  the same passages after the real `alter_facts` perturbation changed one
           number/date. Each repair is scored on whether it undid the change.

Two repairs are tested:
  correct  "fix_alter_facts"       rewrites an out-voted figure, only flags the rest
  drop     "fix_alter_facts_drop"  drops the out-voted / impossible-date passage

It also reports how many altered numbers could be cross-checked at all, i.e. whether another
non-sibling passage states a number with the same signature. This is the key limit: in
HotpotQA most facts appear in only one passage.

Usage (from backend/):
    python test_fix_alter_facts.py
    python test_fix_alter_facts.py --num-questions 50 --seed 42
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from perturbations import apply_context_methods, PerturbationSettings
from repairs import REPAIR_REGISTRY, apply_repair_methods, _fact_signatures, _passages_linked, _title_stem

DATASET_PATH = Path(__file__).parent / "hotpotqa_dataset.json"
VARIANTS = {"correct": "fix_alter_facts", "drop": "fix_alter_facts_drop"}
CONTEXT_SIZES = [2, 3, 5, 10]


def load_dataset():
    if not DATASET_PATH.exists():
        print(f"Could not find {DATASET_PATH}. Run this from inside backend/.")
        sys.exit(1)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_docs(question, k):
    gold = set(question.get("gold_titles") or [])
    docs = [{"title": c["title"], "text": c["text"], "score": c.get("score")}
            for c in question.get("contexts", []) if "Synthetic" not in c["title"]]
    docs.sort(key=lambda d: d["title"] not in gold)  # stable: gold passages first
    return docs[:k]


def crosscheckable(docs, index, offset):
    """True if another, non-sibling passage that refers to this one (or is referred to by it)
    states a number with the same signature."""
    target = next((f for f in _fact_signatures(docs[index]["text"]) if f["start"] == offset), None)
    if target is None:
        return False
    stem = _title_stem(docs[index]["title"])
    for j, doc in enumerate(docs):
        if j == index or _title_stem(doc["title"]) == stem or not _passages_linked(docs[index], doc):
            continue
        if any(f["signature"] == target["signature"] for f in _fact_signatures(doc["text"])):
            return True
    return False


def run_repair(method, docs):
    _, repaired, changes = apply_repair_methods("", [dict(d) for d in docs], [method])
    return repaired, changes[0]


def flagged_titles(record):
    details = record.get("details", {})
    titles = set()
    for item in details.get("flagged", []):
        for names in item["values"].values():
            titles.update(names)
    titles.update(item["title"] for item in details.get("implausible", []))
    return titles


def empty_stats():
    return {"clean_cases": 0, "clean_changed": 0, "clean_flagged": 0, "altered": 0, "crosscheckable": 0,
            "corrected_exactly": 0, "corrected_other": 0, "dropped": 0, "flagged": 0, "missed": 0,
            "collateral": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fix_alter_facts_results.csv")
    args = parser.parse_args()

    for method in VARIANTS.values():
        if method not in REPAIR_REGISTRY:
            print(f"'{method}' is not registered in repairs.py. Paste repairs_alter_facts_addition.py at the end of it.")
            sys.exit(1)

    dataset = load_dataset()
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_questions, len(dataset)))
    settings = PerturbationSettings(fact_count=1)

    stats = {(v, k): empty_stats() for v in VARIANTS for k in CONTEXT_SIZES}
    rows = []
    examples = {}

    for q_index, question in enumerate(sample):
        qid = question.get("id")
        for k in CONTEXT_SIZES:
            docs = pick_docs(question, k)
            if len(docs) < k:
                continue
            altered_docs, changes = apply_context_methods(
                [dict(d) for d in docs], ["alter_facts"], settings, args.seed + q_index, None, None)
            details = changes[0]["details"]
            for variant, method in VARIANTS.items():
                s = stats[(variant, k)]

                # --- clean case: any correction/drop here is a false positive ---
                repaired, record = run_repair(method, docs)
                original_by_title = {d["title"]: d["text"] for d in docs}
                changed = [d for d in repaired if d["text"] != original_by_title.get(d["title"])]
                dropped = len(docs) - len(repaired)
                s["clean_cases"] += 1
                if changed or dropped:
                    s["clean_changed"] += 1
                    examples.setdefault((variant, k, "clean"), record["summary"])
                if flagged_titles(record):
                    s["clean_flagged"] += 1

                # --- altered case ---
                if not details:
                    rows.append({"question_id": qid, "k": k, "variant": variant, "outcome": "no integer token"})
                    continue
                idx = details[0]["index"]
                title = details[0]["title"]
                orig_text = docs[idx]["text"]
                altered_text = altered_docs[idx]["text"]
                checkable = crosscheckable(docs, idx, details[0]["offset"])
                repaired, record = run_repair(method, altered_docs)
                by_title = {d["title"]: d for d in repaired}
                if title not in by_title:
                    outcome = "dropped"
                elif by_title[title]["text"] == orig_text:
                    outcome = "corrected_exactly"
                elif by_title[title]["text"] != altered_text:
                    outcome = "corrected_other"
                elif title in flagged_titles(record):
                    outcome = "flagged"
                else:
                    outcome = "missed"
                collateral = sum(
                    1 for d in docs if d["title"] != title and (
                        d["title"] not in by_title or by_title[d["title"]]["text"] != d["text"]))
                s["altered"] += 1
                s["crosscheckable"] += int(checkable)
                s[outcome] += 1
                s["collateral"] += int(collateral > 0)
                rows.append({"question_id": qid, "k": k, "variant": variant, "outcome": outcome,
                             "crosscheckable": checkable, "collateral_changes": collateral,
                             "original": details[0]["original"], "replacement": details[0]["replacement"]})
                if outcome.startswith("corrected") or outcome == "dropped":
                    examples.setdefault((variant, k, outcome), (title, details[0]["original"], details[0]["replacement"]))

    out = Path(args.output)
    fieldnames = ["question_id", "k", "variant", "outcome", "crosscheckable", "collateral_changes",
                  "original", "replacement"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = ("| Repair | Passages per case | Altered cases | Cross-checkable | Corrected exactly | "
              "Altered passage dropped | Flagged only | Missed | Other passages wrongly changed | "
              "Clean cases wrongly changed | Clean cases flagged |")
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for variant, method in VARIANTS.items():
        for k in CONTEXT_SIZES:
            s = stats[(variant, k)]
            lines.append(
                f"| {method} | {k} | {s['altered']} | {s['crosscheckable']} | {s['corrected_exactly']} | "
                f"{s['dropped']} | {s['flagged']} | {s['missed']} | {s['collateral']} | "
                f"{s['clean_changed']}/{s['clean_cases']} | {s['clean_flagged']}/{s['clean_cases']} |")
    table = "\n".join(lines)
    print("\n" + table)

    example_lines = []
    for (variant, k, kind), value in sorted(examples.items(), key=lambda kv: str(kv[0])):
        example_lines.append(f"- `{VARIANTS[variant]}`, {k} passages, {kind}: {value}")
    examples_text = "\n".join(example_lines) or "- None observed in this run."

    total_altered = stats[("correct", 10)]["altered"]
    findings = Path("fix_alter_facts_findings.md")
    with open(findings, "w", encoding="utf-8") as f:
        f.write(f"""# Altered-Facts Repair: Detection Findings (RS-62)

**Methods under test (backend/repairs.py):**
- `fix_alter_facts`: cross-checks numbers across passages, corrects a figure the other passages out-vote, flags the rest
- `fix_alter_facts_drop`: drops the passage whose figure is out-voted or contains an impossible date

**Sample:** {len(sample)} questions, seed {args.seed}. For each passage-set size the real `alter_facts` perturbation
changed one number or date ({total_altered} altered cases at 10 passages). Passages are the question's own candidate
passages, gold passages first.

## Results

{table}

- **Cross-checkable** counts altered numbers for which another non-sibling passage that refers to this one (its
  text mentions the other's title) states a number with the same signature (same two preceding words, next word
  and size class). Only these can ever be fixed.
- **Corrected exactly** means the repaired passage text equals the original text before alteration.
- **Clean cases wrongly changed** counts cases where the repair corrected or dropped a passage although nothing was altered.

## What the results say

- **It only works where a fact is repeated.** In HotpotQA a fact usually appears in one passage, so most altered
  numbers have nothing to cross-check against (see the Cross-checkable column). When there is no second passage,
  the repair cannot know the original value.
- **A majority is needed to correct.** With a one-against-one disagreement the repair cannot tell which passage is
  right, so it only flags it. Corrections need at least 2 passages agreeing against 1, which means 3 or more
  retrieved passages. The default top-k of 2 almost never allows it.
- **Impossible dates are the one signal that needs no second passage.** A date like 30 February is flagged (or the
  passage dropped), but the original value cannot be recovered.
- **False positives come from sibling passages.** Passages about similar entities (for example consecutive years of
  an event) can contain the same sentence with different numbers. Titles that differ only by digits are skipped, but
  other look-alike passages can still trigger a wrong correction.

## Examples

{examples_text}

## Limitations

- The signature matching is lexical. The same fact worded differently in two passages is not matched.
- Numbers are matched only if they are bounded integers (the same kind `alter_facts` changes). Decimals and numbers
  with thousands separators are ignored.
- Dropping a passage removes all of its evidence, not just the wrong figure, so `fix_alter_facts_drop` can hurt answers.
- This is an offline detection test. Whether answers improve is measured separately with the dashboard pipeline.

## Suggested next steps

- Use a semantic or entity-aware matcher so reworded facts can be compared.
- Retrieve more passages (higher top-k) when a figure disagrees, to obtain a majority.
- Verify disputed figures against a trusted source instead of only against other retrieved passages.
""")
    print(f"\nPer-case results: {out.resolve()}")
    print(f"Findings:         {findings.resolve()}")


if __name__ == "__main__":
    main()
