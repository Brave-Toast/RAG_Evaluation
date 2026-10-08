"""
Accuracy test for repairs.py's detector filters (RS-57).
------------------------------------------------------------------
Ticket: Conflicting-Context Repair Method (Contradiction Detector + Filter)
"Detect fake contradicting documents and filter them out. Test detector
accuracy. Write up findings."

Runs entirely offline (no server, no LLM, no embeddings).

Two filter variants are tested:
  full       -> "contradiction_filter"            (title prefix + text markers)
  text_only  -> "contradiction_filter_text_only"  (text markers only; no label)

For each sampled question, these cases are built:
  clean                 real contexts only. Any removal = false positive.
  conflict              real contexts + fake conflicting doc (same wording as
                        app.py's inject_conflicting_context).
  paraphrased_conflict  real contexts + a fake conflicting doc that avoids the
                        filter's keywords and the "Synthetic" title. Shows what
                        the filter misses.
  metadata_injection    real contexts + metadata-boosted distractor.

Plus one small run of "adversarial_clean": genuine-looking passages that
legitimately use words like "refute that" or "alternative facts", and one with
a title starting with "Synthetic" (e.g. "Synthetic biology"). All should be KEPT.

Usage:
    cd backend
    python test_repair_accuracy.py --num-questions 20
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from perturbations import (
    build_metadata_boosted_candidate,
    distractor_pool,
    method_rng,
    PerturbationSettings,
)
from repairs import apply_repair_methods, REPAIR_REGISTRY


DATASET_PATH = Path(__file__).parent / "hotpotqa_dataset.json"

VARIANTS = {
    "full": "contradiction_filter",
    "text_only": "contradiction_filter_text_only",
}

CASE_ORDER = ["clean", "conflict", "paraphrased_conflict", "metadata_injection", "adversarial_clean"]
ORIGINAL_CASES = ["clean", "conflict", "metadata_injection"]  # the original 20 x 3 test

# Genuine-style passages that a naive filter might wrongly remove.
ADVERSARIAL_DOCS = [
    {"title": "Synthetic biology",
     "text": "Synthetic biology is an interdisciplinary field that designs and builds new biological parts, devices and systems."},
    {"title": "Cold fusion controversy",
     "text": "In 1989 two chemists claimed to observe nuclear fusion at room temperature. Later experiments failed to replicate the result, and many physicists set out to refute that claim."},
    {"title": "Post-truth politics",
     "text": "The phrase alternative facts entered public debate in 2017 after a press briefing, and commentators criticised it as a way of dismissing verifiable reporting."},
]

# Paraphrased conflict wordings. None contain: "conflicting modern sources",
# "refute that", "alternative facts", or "[Topic tags:".
PARAPHRASE_TEMPLATES = [
    "Recent scholarship on '{q}' suggests the commonly cited answer, '{gt}', is mistaken. Newly reviewed records point to a different conclusion.",
    "A 2019 re-examination of the evidence behind '{q}' found that '{gt}' does not hold up. Researchers now favour another answer.",
    "Editors have noted that the widely repeated claim '{gt}' in reply to '{q}' was based on an error and has since been corrected.",
]


def load_dataset():
    if not DATASET_PATH.exists():
        print(f"Could not find {DATASET_PATH}. Run this from inside the backend/ folder.")
        sys.exit(1)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_conflict_doc(query, ground_truth):
    """Mirrors app.py's inject_conflicting_context, kept local so this script
    doesn't need to import the FastAPI app."""
    return {
        "title": "Synthetic_Conflicting_Record",
        "text": (
            f"NOTICE: Regarding the query '{query}', conflicting modern sources "
            f"refute that '{ground_truth}' is accurate. Historical logs indicate "
            f"synthetic alternative facts."
        ),
        "score": 0.99,
    }


def build_paraphrased_conflict_doc(query, ground_truth, index):
    template = PARAPHRASE_TEMPLATES[index % len(PARAPHRASE_TEMPLATES)]
    return {
        "title": "Regional_Archive_Note",
        "text": template.format(q=query, gt=ground_truth),
        "score": 0.99,
    }


def score_case(contexts_before, contexts_after, injected_title):
    """Returns (counts, missed_docs, wrongly_removed_titles).
    tp = injected doc removed, fn = injected doc kept,
    fp = genuine doc removed, tn = genuine doc kept."""
    kept_titles = {d["title"] for d in contexts_after}
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    missed, wrongly_removed = [], []
    for doc in contexts_before:
        is_injected = injected_title is not None and doc["title"] == injected_title
        was_kept = doc["title"] in kept_titles
        if is_injected:
            if was_kept:
                counts["fn"] += 1
                missed.append(doc)
            else:
                counts["tp"] += 1
        else:
            if was_kept:
                counts["tn"] += 1
            else:
                counts["fp"] += 1
                wrongly_removed.append(doc["title"])
    return counts, missed, wrongly_removed


def safe_div(a, b):
    return a / b if b else None


def metrics(c):
    precision = safe_div(c["tp"], c["tp"] + c["fp"])
    recall = safe_div(c["tp"], c["tp"] + c["fn"])
    return precision, recall


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def empty_counts():
    return {"tp": 0, "fn": 0, "fp": 0, "tn": 0}


def add_counts(total, counts):
    for key in total:
        total[key] += counts[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="repair_accuracy_results.csv")
    args = parser.parse_args()

    for method in VARIANTS.values():
        if method not in REPAIR_REGISTRY:
            print(f"'{method}' is not registered in repairs.py. Add the text-only filter first (see instructions).")
            sys.exit(1)

    dataset = load_dataset()
    if not dataset:
        print("Dataset is empty.")
        sys.exit(1)

    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_questions, len(dataset)))
    settings = PerturbationSettings()

    rows = []
    stats = {(v, c): empty_counts() for v in VARIANTS for c in CASE_ORDER}
    examples_missed = {}   # (variant, case) -> first missed injected doc
    examples_removed = {}  # (variant, case) -> first wrongly removed title

    def run_case(variant, case, question_id, query, contexts, injected_title):
        method = VARIANTS[variant]
        _, filtered, _ = apply_repair_methods(query, list(contexts), [method])
        counts, missed, wrongly_removed = score_case(contexts, filtered, injected_title)
        add_counts(stats[(variant, case)], counts)
        notes = []
        if missed:
            notes.append("injected doc missed")
            examples_missed.setdefault((variant, case), missed[0])
        if wrongly_removed:
            notes.append("genuine doc removed: " + ", ".join(wrongly_removed))
            examples_removed.setdefault((variant, case), wrongly_removed[0])
        rows.append({"question_id": question_id, "variant": variant, "case": case,
                     **counts, "note": "; ".join(notes)})

    for index, question in enumerate(sample):
        qid = question.get("id")
        query = question["question"]
        ground_truth = question["ground_truth"]
        real_contexts = [
            {"title": c["title"], "text": c["text"], "score": c.get("score")}
            for c in question.get("contexts", [])
        ]

        cases = [("clean", real_contexts, None)]

        conflict_doc = build_conflict_doc(query, ground_truth)
        cases.append(("conflict", [conflict_doc] + real_contexts, conflict_doc["title"]))

        para_doc = build_paraphrased_conflict_doc(query, ground_truth, index)
        cases.append(("paraphrased_conflict", [para_doc] + real_contexts, para_doc["title"]))

        pool = distractor_pool(dataset, qid, real_contexts, query)
        rng = method_rng(args.seed, "metadata_injection")
        candidate, _ = build_metadata_boosted_candidate(query, pool, real_contexts, settings, rng)
        if candidate is not None:
            cases.append(("metadata_injection", [candidate] + real_contexts, candidate["title"]))
        else:
            for variant in VARIANTS:
                rows.append({"question_id": qid, "variant": variant, "case": "metadata_injection",
                             "tp": 0, "fn": 0, "fp": 0, "tn": 0, "note": "no distractor available"})

        for variant in VARIANTS:
            for case, contexts, injected_title in cases:
                run_case(variant, case, qid, query, contexts, injected_title)

    # Small false-positive test on genuine passages that look suspicious.
    for variant in VARIANTS:
        run_case(variant, "adversarial_clean", "n/a", "", list(ADVERSARIAL_DOCS), None)

    # ---- Save per-case CSV ----
    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question_id", "variant", "case", "tp", "fn", "fp", "tn", "note"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-case results to {output_path.resolve()}\n")

    # ---- Summary table (printed + written to markdown) ----
    table_lines = ["| Filter | Case | TP | FN | FP | TN | Precision | Recall |",
                   "|---|---|---|---|---|---|---|---|"]
    combined = {}
    for variant in VARIANTS:
        original = empty_counts()
        for case in ORIGINAL_CASES:
            add_counts(original, stats[(variant, case)])
        combined[variant] = original
        for case in CASE_ORDER + ["original_test"]:
            c = original if case == "original_test" else stats[(variant, case)]
            p, r = metrics(c)
            label = "original test (clean+conflict+metadata)" if case == "original_test" else case
            table_lines.append(
                f"| {variant} | {label} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} | {fmt(p)} | {fmt(r)} |"
            )
    table = "\n".join(table_lines)
    print(table)

    # ---- Findings ----
    full_para = stats[("full", "paraphrased_conflict")]
    text_para = stats[("text_only", "paraphrased_conflict")]
    full_adv = stats[("full", "adversarial_clean")]
    text_adv = stats[("text_only", "adversarial_clean")]
    _, full_para_recall = metrics(full_para)
    _, text_para_recall = metrics(text_para)
    n_pos = combined["full"]["tp"] + combined["full"]["fn"]
    n_docs = sum(combined["full"].values())

    failure_lines = []
    for (variant, case), doc in examples_missed.items():
        failure_lines.append(f"- **Missed ({variant} filter, {case}):** title `{doc['title']}`, text: \"{doc['text']}\"")
    for (variant, case), title in examples_removed.items():
        failure_lines.append(f"- **Wrongly removed ({variant} filter, {case}):** a genuine document titled `{title}`")
    failures = "\n".join(failure_lines) if failure_lines else "- None observed in this run."

    ci_note = ""
    if combined["full"]["fn"] == 0 and n_pos:
        ci_note = (f" With 0 misses out of {n_pos} injected documents, the true recall is only known to be "
                   f"at least about {1 - 3 / n_pos:.0%} (95% confidence), not exactly 100%.")

    findings_path = Path("repair_accuracy_findings.md")
    with open(findings_path, "w", encoding="utf-8") as f:
        f.write(f"""# Conflicting-Context Repair Method — Detector Accuracy Findings

**Methods under test (backend/repairs.py):**
- `contradiction_filter` ("full": title prefix + text markers)
- `contradiction_filter_text_only` ("text_only": text markers only, no title label)

**Sample:** {len(sample)} questions, seed {args.seed}. The original test (clean / conflict / metadata_injection)
covers {n_docs} documents, of which {n_pos} are injected. The paraphrased-conflict case and a
{len(ADVERSARIAL_DOCS)}-document adversarial false-positive set were added on top.

## Results

{table}

## What the results say

- **The original test is easy.** The detector looks for exact strings that our own
  generators insert (the `Synthetic` title, `conflicting modern sources`, `refute that`,
  `alternative facts`, `[Topic tags:`), so a near-perfect score on that test mainly shows
  the filter matches our own generators.{ci_note}
- **Paraphrased conflicts are the realistic test.** Recall on conflicting documents that avoid those
  keywords was {fmt(full_para_recall)} (full filter) and {fmt(text_para_recall)} (text-only filter).
- **False positives on genuine text:** on {len(ADVERSARIAL_DOCS)} genuine passages the full filter wrongly removed
  {full_adv['fp']} and the text-only filter wrongly removed {text_adv['fp']}. The title rule
  (`startswith("synthetic")`) can remove real articles such as "Synthetic biology".
- **Metadata injection is a different perturbation.** It adds an off-topic passage made to look relevant,
  not a contradiction, so it is reported on its own row.

## Failure examples

{failures}

## How the detector works

1. **Title prefix "Synthetic"** — our own internal naming convention for test documents (a label, not content).
2. **Suspicious phrasing in the text** — regex for wording our perturbations insert.

## Limitations

- This is a lexical/pattern-based detector, not a semantic one. It did not detect
  reworded conflicting documents (see the paraphrased_conflict rows).
- The marker list is tied to our own injection formats in `perturbations.py` and `app.py`;
  if those change, the detector must change too.
- The adversarial false-positive set is small and hand-written, so it shows failure modes
  rather than a reliable false-positive rate.

## Suggested next steps

- Add a semantic/NLI-based second layer to catch reworded conflicting content.
- Build a larger false-positive set from real passages about disputes, corrections and retractions.
- Replace the title rule with a content-based check, since the prefix can match real articles.
""")
    print(f"\nFindings written to {findings_path.resolve()}")


if __name__ == "__main__":
    main()