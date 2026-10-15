"""Evaluate answer accuracy for clean, typo-broken, and spelling-fixed RAG.

Correctness follows the project's ``diagnose_failure`` rule: normalized token
F1 must be at least 0.5. Normalization lowercases, removes punctuation, and
collapses whitespace. Exact match after the same normalization is also stored.

The runner uses the same generation model and prompt mode for all conditions.
It rejects synthetic fallback answers, checkpoints after every question, and
can resume an interrupted run from its output file.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from .app import ContextDoc, generate_llm_answer, inject_spelling_typos, perform_vector_retrieval
from .isolation_metrics import calculate_exact_match, calculate_token_f1
from .repairs import apply_repair_methods

DATASET = Path(__file__).with_name("hotpotqa_dataset.json")
F1_CORRECT_THRESHOLD = 0.5


def _retrieval_result(docs):
    return [{"title": doc.title, "score": doc.score} for doc in docs]


def _score(answer, expected):
    f1 = calculate_token_f1(answer, expected)
    exact_match = bool(calculate_exact_match(answer, expected))
    return {"answer": answer, "token_f1": f1, "exact_match": exact_match,
            "correct": f1 >= F1_CORRECT_THRESHOLD}


def summarize(results):
    total = len(results)
    counts = {
        name: sum(bool(row[name]["correct"]) for row in results)
        for name in ("clean", "broken", "fixed")
    }
    recovered = sum(not row["broken"]["correct"] and row["fixed"]["correct"] for row in results)
    worsened = sum(row["broken"]["correct"] and not row["fixed"]["correct"] for row in results)
    broken_incorrect = total - counts["broken"]
    return {
        "questions_completed": total,
        "accuracy": {name: {"correct": count, "total": total,
                             "fraction": count / total if total else None}
                     for name, count in counts.items()},
        "broken_incorrect_recovered_by_fixed": recovered,
        "broken_correct_worsened_by_fixed": worsened,
        "overall_recovery_percentage": (100.0 * recovered / broken_incorrect)
        if broken_incorrect else 0.0,
        "recovery_denominator_broken_incorrect": broken_incorrect,
    }


def _write_checkpoint(path, config, results, complete=False):
    payload = {
        "complete": complete,
        "scoring": {
            "correctness": "normalized token F1 >= 0.5",
            "normalization": "lowercase, remove punctuation, collapse whitespace",
            "exact_match": "stored separately after the same normalization",
        },
        "config": config,
        "summary": summarize(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _generate(query, contexts, model, expected):
    answer, warning = await generate_llm_answer(
        query=query, contexts=contexts, model_choice=model, ground_truth=expected,
        is_perturbed=True, perturbation_type="spelling",
    )
    if warning:
        raise RuntimeError(
            f"Generation model did not produce a genuine answer ({warning}). "
            "Synthetic fallback output was discarded."
        )
    return answer


async def run(args):
    if args.model == "synthetic-evaluator":
        raise ValueError("synthetic-evaluator cannot measure generated-answer accuracy")
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.questions != len(rows):
        raise ValueError(f"This benchmark requires all {len(rows)} questions; got {args.questions}")

    config = {
        "questions": args.questions, "seed": args.seed, "typo_rate": args.typo_rate,
        "top_k": args.top_k, "search_type": args.search_type,
        "embedding_provider": args.embedding_provider, "generation_model": args.model,
        "f1_correct_threshold": F1_CORRECT_THRESHOLD,
    }
    results = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError("Existing checkpoint configuration differs; choose a new output path")
        results = existing.get("results", [])

    for index, row in enumerate(rows[len(results):], start=len(results)):
        original = row["question"]
        typo = inject_spelling_typos(original, args.typo_rate, random.Random(args.seed + index))
        fixed, _, repair_changes = apply_repair_methods(
            typo, [], ["fix_spelling"], {"protected_entities": row.get("gold_titles", [])}
        )
        candidates = [ContextDoc(**doc) for doc in row["contexts"]]
        queries = {"clean": original, "broken": typo, "fixed": fixed}
        retrieved = {
            name: perform_vector_retrieval(
                candidates, query, args.search_type, args.top_k, args.embedding_provider
            ) for name, query in queries.items()
        }
        expected = row["ground_truth"]
        answers = {}
        for name in ("clean", "broken", "fixed"):
            generated = await _generate(queries[name], retrieved[name], args.model, expected)
            answers[name] = {
                "query": queries[name], "retrieval": _retrieval_result(retrieved[name]),
                **_score(generated, expected),
            }
        answers["fixed"]["repair_changes"] = repair_changes
        answers["broken"]["repair_changes"] = []
        answers["clean"]["repair_changes"] = []
        results.append({"id": row["id"], "expected_answer": expected, **answers})
        _write_checkpoint(args.output, config, results)
        print(f"[{index + 1}/{len(rows)}] {row['id']} complete")

    _write_checkpoint(args.output, config, results, complete=True)
    print(json.dumps(summarize(results), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--typo-rate", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--search-type", choices=["similarity", "mmr"], default="similarity")
    parser.add_argument("--embedding-provider", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--model", default="ollama/qwen3.5:9b")
    parser.add_argument("--output", type=Path, default=Path("results/fix_spelling_answers_80.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
