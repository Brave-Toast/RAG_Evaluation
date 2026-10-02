"""
Cross-embedding batch for RS-46: runs the same perturbations through the LIVE backend with
mpnet and nomic-embed-text, then compares the two.

Why through the live API: the server appends every run to stage_run_logs.jsonl, so this also
produces the "batch of perturbation test cases using nomic-embed-text" as log evidence.

Before running:
  1. Start the backend:        cd backend && python app.py
  2. Start Ollama with nomic:  ollama pull nomic-embed-text   (Ollama must be running)

Usage (from backend/, in a second terminal):
    python run_nomic_batch.py                       # 10 questions x 4 methods x 2 models
    python run_nomic_batch.py --num-questions 10 --top-k 2
    python run_nomic_batch.py --llm ollama/qwen3.5:9b   # real LLM answers (slow)

By default the LLM is "synthetic-evaluator" (scripted answers), because this test is about
retrieval. Only the retrieval columns are meaningful in that mode.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"
DATASET_PATH = Path(__file__).parent / "hotpotqa_dataset.json"
PROVIDERS = ["sentence-transformers/all-mpnet-base-v2", "ollama/nomic-embed-text"]
METHODS = ["spelling", "emoji", "conflict", "metadata_injection"]
SYNTHETIC_METHODS = {"conflict", "metadata_injection"}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm", default="synthetic-evaluator")
    parser.add_argument("--providers", nargs="+", default=PROVIDERS)
    args = parser.parse_args()

    try:
        health = httpx.get(f"{API}/api/health", timeout=5.0).json()
    except Exception as exc:
        print(f"Backend is not reachable at {API} ({exc}). Start it with: python app.py")
        sys.exit(1)
    if not health.get("ollama_support", True) and any(p.startswith("ollama/") for p in args.providers):
        print("Warning: backend reports Ollama is not reachable; nomic runs will fail.")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_questions, len(dataset)))

    rows, errors = [], {}
    for provider in args.providers:
        failures = 0
        for i, q in enumerate(sample):
            payload = {
                "question_id": q.get("id"),
                "baseline_query": q["question"],
                "ground_truth": q["ground_truth"],
                "contexts": [{"title": c["title"], "text": c["text"]} for c in q.get("contexts", [])],
                "gold_titles": q.get("gold_titles", []),
                "perturbation_types": METHODS,
                "mode": "separate",
                "settings": {},
                "seed": args.seed + i,
                "typo_rate": 0.25,
                "embedding_provider": provider,
                "search_type": "similarity",
                "top_k": args.top_k,
                "llm_model": args.llm,
            }
            print(f"[{provider}] question {i + 1}/{len(sample)}")
            try:
                res = httpx.post(f"{API}/api/run-evaluation", json=payload, timeout=900.0)
                res.raise_for_status()
                data = res.json()
            except Exception as exc:
                failures += 1
                errors.setdefault(provider, str(exc))
                print(f"  failed: {exc}")
                if failures >= 2:
                    print(f"  stopping {provider} after 2 failures")
                    break
                continue
            for variant in data["variants"]:
                metrics = variant["metrics"]
                retrieved_titles = [c["title"] for c in variant.get("perturbed_retrieved_contexts", [])]
                rows.append({
                    "provider": provider,
                    "question_id": q.get("id"),
                    "method": "+".join(variant["methods"]),
                    "query_embedding_shift": metrics["query_embedding_shift"],
                    "retrieved_overlap_jaccard": metrics["retrieval_rank_shift_jaccard"],
                    "gold_recall_baseline": metrics["context_recall_baseline"],
                    "gold_recall_perturbed": metrics["context_recall_perturbed"],
                    "synthetic_doc_retrieved": any(t.startswith("Synthetic") for t in retrieved_titles),
                })

    if not rows:
        print("No results were collected.")
        sys.exit(1)

    csv_path = Path("cross_embedding_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| Embedding model | Method | Cases | Query shift | Retrieved overlap | "
             "Gold recall (clean -> perturbed) | Injected doc retrieved |",
             "|---|---|---|---|---|---|---|"]
    for provider in args.providers:
        for method in METHODS:
            group = [r for r in rows if r["provider"] == provider and r["method"] == method]
            if not group:
                continue
            injected = (f"{sum(r['synthetic_doc_retrieved'] for r in group)}/{len(group)}"
                        if method in SYNTHETIC_METHODS else "-")
            lines.append(
                f"| {provider} | {method} | {len(group)} | "
                f"{fmt(mean([r['query_embedding_shift'] for r in group]))} | "
                f"{fmt(mean([r['retrieved_overlap_jaccard'] for r in group]))} | "
                f"{fmt(mean([r['gold_recall_baseline'] for r in group]))} -> "
                f"{fmt(mean([r['gold_recall_perturbed'] for r in group]))} | {injected} |")
    table = "\n".join(lines)
    print("\n" + table)

    error_text = "\n".join(f"- `{p}` had errors: {e}" for p, e in errors.items()) or "- None."
    md_path = Path("cross_embedding_batch_findings.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Cross-Embedding Batch: mpnet vs nomic-embed-text

{len(sample)} questions, seed {args.seed}, top-k {args.top_k}, LLM `{args.llm}`.
Every method runs separately from the clean input. All runs are also logged in `stage_run_logs.jsonl`.

{table}

- **Query shift:** cosine similarity between the original and perturbed question embeddings (1.00 = unchanged).
- **Retrieved overlap:** Jaccard overlap between the clean and perturbed retrieved document sets.
- **Injected doc retrieved:** how often the synthetic document reached the top-k (conflict and metadata injection only).
{"- Answer-based metrics are not meaningful when the LLM is the scripted `synthetic-evaluator`." if args.llm == "synthetic-evaluator" else ""}

## Errors

{error_text}
""")
    print(f"\nPer-run CSV: {csv_path.resolve()}")
    print(f"Findings:    {md_path.resolve()}")


if __name__ == "__main__":
    main()
