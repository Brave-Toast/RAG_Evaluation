"""
Retrieval test for the metadata_injection perturbation (RS-44, "retrieves the wrong chunk on 10 cases").

For each sampled question and each embedding provider it:
  1. retrieves top-k from the CLEAN contexts (baseline),
  2. builds the metadata-boosted off-topic candidate, adds it to the same contexts,
     and retrieves top-k again,
  3. records whether the injected chunk reached the top-k, its rank, and whether any
     gold passage was pushed out.

Needs no LLM. Embedding providers are checked before use, so an "ollama/nomic-embed-text"
run can never silently be MiniLM (app.compute_dense_embedding falls back without telling you).

Usage (from backend/):
    python test_metadata_injection.py                          # 10 questions, mpnet + nomic
    python test_metadata_injection.py --num-questions 10 --top-k 2
    python test_metadata_injection.py --providers sentence-transformers/all-mpnet-base-v2

Run it once BEFORE pasting perturbations_metadata_patch.py and once AFTER, and keep both
outputs (use --tag before / --tag after) to show the improvement.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import httpx

from perturbations import (
    build_metadata_boosted_candidate,
    distractor_pool,
    method_rng,
    PerturbationSettings,
)
from app import (
    ContextDoc,
    perform_vector_retrieval,
    ensure_ollama_model_pulled,
    get_sentence_transformer,
)

DATASET_PATH = Path(__file__).parent / "hotpotqa_dataset.json"
DEFAULT_PROVIDERS = [
    "sentence-transformers/all-mpnet-base-v2",
    "ollama/nomic-embed-text",
]


def load_dataset():
    if not DATASET_PATH.exists():
        print(f"Could not find {DATASET_PATH}. Run this from inside backend/.")
        sys.exit(1)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def provider_ok(provider):
    """True only if this exact model can really be used (no silent fallback)."""
    if provider.startswith("ollama/"):
        name = provider.split("/", 1)[1]
        if not ensure_ollama_model_pulled(name):
            return False, f"Ollama model '{name}' is not available (is Ollama running?)"
        try:
            res = httpx.post("http://localhost:11434/api/embeddings",
                             json={"model": name, "prompt": "ping"}, timeout=120.0)
            if res.status_code == 200 and res.json().get("embedding"):
                return True, ""
            return False, f"Ollama embeddings returned HTTP {res.status_code}"
        except Exception as exc:
            return False, f"Ollama embeddings call failed: {exc}"
    if provider.startswith("openai/"):
        return False, "OpenAI provider is not supported by this test"
    if get_sentence_transformer(provider.replace("huggingface/", "")) is None:
        return False, "SentenceTransformer model could not be loaded"
    return True, ""


def recall(gold, titles):
    if not gold:
        return None
    return len(set(gold) & set(titles)) / len(set(gold))


def run_provider(provider, sample, dataset, args, settings):
    rows = []
    for q in sample:
        qid = q.get("id")
        query = q["question"]
        real = [{"title": c["title"], "text": c["text"], "score": c.get("score")}
                for c in q.get("contexts", []) if "Synthetic" not in c["title"]]
        gold = q.get("gold_titles") or q.get("supporting_titles") or []

        pool = distractor_pool(dataset, qid, real, query)
        rng = method_rng(args.seed, "metadata_injection")
        cand, _ = build_metadata_boosted_candidate(query, pool, real, settings, rng)
        if cand is None:
            rows.append({"provider": provider, "question_id": qid, "status": "skipped (no candidate)"})
            continue

        clean_docs = [ContextDoc(**d) for d in real]
        base = perform_vector_retrieval(list(clean_docs), query, args.search_type, args.top_k, provider)
        pert = perform_vector_retrieval(
            [ContextDoc(title=cand["title"], text=cand["text"], score=None)] + list(clean_docs),
            query, args.search_type, args.top_k, provider)

        base_titles = [d.title for d in base]
        pert_titles = [d.title for d in pert]
        rank = pert_titles.index(cand["title"]) + 1 if cand["title"] in pert_titles else None
        displaced = [t for t in gold if t in base_titles and t not in pert_titles]
        rows.append({
            "provider": provider,
            "question_id": qid,
            "status": "ok",
            "query": query,
            "injected_title": cand["title"],
            "injected_in_topk": rank is not None,
            "injected_rank": rank,
            "gold_recall_baseline": recall(gold, base_titles),
            "gold_recall_perturbed": recall(gold, pert_titles),
            "gold_displaced": "; ".join(displaced),
            "baseline_topk": " | ".join(base_titles),
            "perturbed_topk": " | ".join(pert_titles),
        })
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def summarise(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    n = len(ok)
    hits = sum(1 for r in ok if r["injected_in_topk"])
    top1 = sum(1 for r in ok if r["injected_rank"] == 1)
    displaced = sum(1 for r in ok if r["gold_displaced"])
    return {
        "n": n, "hits": hits, "top1": top1, "displaced": displaced,
        "recall_base": mean([r["gold_recall_baseline"] for r in ok]),
        "recall_pert": mean([r["gold_recall_perturbed"] for r in ok]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-type", default="similarity", choices=["similarity", "mmr"])
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS)
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. before / after")
    args = parser.parse_args()

    dataset = load_dataset()
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_questions, len(dataset)))
    settings = PerturbationSettings()
    suffix = f"_{args.tag}" if args.tag else ""

    all_rows, summaries, skipped = [], {}, {}
    for provider in args.providers:
        ok, reason = provider_ok(provider)
        if not ok:
            skipped[provider] = reason
            print(f"SKIPPING {provider}: {reason}")
            continue
        print(f"Running {provider} on {len(sample)} questions...")
        rows = run_provider(provider, sample, dataset, args, settings)
        all_rows.extend(rows)
        summaries[provider] = summarise(rows)

    if not all_rows:
        print("No provider could be run.")
        sys.exit(1)

    csv_path = Path(f"metadata_injection_results{suffix}.csv")
    fields = ["provider", "question_id", "status", "query", "injected_title", "injected_in_topk",
              "injected_rank", "gold_recall_baseline", "gold_recall_perturbed", "gold_displaced",
              "baseline_topk", "perturbed_topk"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    lines = ["| Embedding model | Cases | Injected chunk in top-k | Injected at rank 1 | "
             "Gold passage pushed out | Gold recall (clean -> injected) |",
             "|---|---|---|---|---|---|"]
    for provider, s in summaries.items():
        lines.append(f"| {provider} | {s['n']} | {s['hits']}/{s['n']} | {s['top1']}/{s['n']} | "
                     f"{s['displaced']}/{s['n']} | {fmt(s['recall_base'])} -> {fmt(s['recall_pert'])} |")
    table = "\n".join(lines)
    print("\n" + table)

    case_lines = ["| Embedding model | Question | Injected rank | Top-k after injection |", "|---|---|---|---|"]
    for r in all_rows:
        if r["status"] != "ok":
            continue
        q = r["query"] if len(r["query"]) <= 60 else r["query"][:57] + "..."
        case_lines.append(f"| {r['provider']} | {q} | {r['injected_rank'] or 'not retrieved'} | {r['perturbed_topk']} |")
    cases = "\n".join(case_lines)

    skipped_text = "\n".join(f"- `{p}` was skipped: {why}" for p, why in skipped.items()) or "- None."
    md_path = Path(f"metadata_injection_findings{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Metadata Injection: Retrieval Test Findings

**Perturbation:** `metadata_injection` (backend/perturbations.py), which adds an off-topic passage
carrying query keywords before retrieval.
**Setup:** {len(sample)} questions, seed {args.seed}, top-k {args.top_k}, search type `{args.search_type}`.
Each case retrieves from the clean contexts, then again with the injected chunk added.

## Results

{table}

- **Injected chunk in top-k** is the pass criterion for "retrieves the wrong chunk".
- **Gold passage pushed out** counts cases where a gold passage retrieved on the clean run
  disappeared from the top-k after injection.

## Per-case results

{cases}

## Providers not run

{skipped_text}

## Notes and limits

- Retrieval embeds the passage text only (the title is metadata on the Chroma path), so the
  injected keywords sit in the text.
- Each question has about 10 candidate passages, so top-k is a small window. Results on a
  larger candidate pool may differ.
- {len(sample)} cases is a small sample. Treat the rates as indicative, not precise.
""")
    print(f"\nPer-case CSV: {csv_path.resolve()}")
    print(f"Findings:     {md_path.resolve()}")


if __name__ == "__main__":
    main()
