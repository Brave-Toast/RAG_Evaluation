"""Reproducible typo retrieval benchmark; it does not generate answers."""

import argparse
import json
import random
from pathlib import Path

from .app import ContextDoc, inject_spelling_typos, perform_vector_retrieval
from .isolation_metrics import calculate_context_recall
from .repairs import apply_repair_methods

DATASET = Path(__file__).with_name("hotpotqa_dataset.json")


def _titles(docs):
    return [doc.title for doc in docs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--typo-rate", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--search-type", choices=["similarity", "mmr"], default="similarity")
    parser.add_argument("--embedding-provider", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output", type=Path, default=Path("results/fix_spelling_80.json"))
    args = parser.parse_args(argv)

    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.questions < 1 or args.questions > len(rows):
        parser.error(f"--questions must be between 1 and {len(rows)}")

    results = []
    for index, row in enumerate(rows[:args.questions]):
        original = row["question"]
        typo = inject_spelling_typos(original, args.typo_rate, random.Random(args.seed + index))
        fixed, _, changes = apply_repair_methods(
            typo, [], ["fix_spelling"], {"protected_entities": row.get("gold_titles", [])}
        )
        docs = [ContextDoc(**doc) for doc in row["contexts"]]
        clean_docs = perform_vector_retrieval(docs, original, args.search_type, args.top_k, args.embedding_provider)
        typo_docs = perform_vector_retrieval(docs, typo, args.search_type, args.top_k, args.embedding_provider)
        fixed_docs = perform_vector_retrieval(docs, fixed, args.search_type, args.top_k, args.embedding_provider)
        gold = row.get("gold_titles", [])
        results.append({
            "id": row["id"], "original_query": original, "typo_query": typo, "fixed_query": fixed,
            "exact_query_recovery": fixed == original, "repair": changes[0],
            "clean_titles": _titles(clean_docs), "typo_titles": _titles(typo_docs), "fixed_titles": _titles(fixed_docs),
            "clean_context_recall": calculate_context_recall(_titles(clean_docs), gold),
            "typo_context_recall": calculate_context_recall(_titles(typo_docs), gold),
            "fixed_context_recall": calculate_context_recall(_titles(fixed_docs), gold),
        })

    summary = {
        "questions": len(results), "seed": args.seed, "typo_rate": args.typo_rate,
        "top_k": args.top_k, "search_type": args.search_type,
        "embedding_provider": args.embedding_provider,
        "exact_query_recoveries": sum(r["exact_query_recovery"] for r in results),
        "mean_clean_context_recall": sum(r["clean_context_recall"] for r in results) / len(results),
        "mean_typo_context_recall": sum(r["typo_context_recall"] for r in results) / len(results),
        "mean_fixed_context_recall": sum(r["fixed_context_recall"] for r in results) / len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote per-question results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
