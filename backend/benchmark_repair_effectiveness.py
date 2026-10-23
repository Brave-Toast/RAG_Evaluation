"""Benchmark matched perturbation repairs using genuine LLM-as-Judge scores.

Each perturbation/repair pair is evaluated twice with identical question,
seed, and backend settings: once without the repair and once with it.  The
per-question CSV is checkpointed after every completed pair and may be
resumed.  ``--no-resume`` moves existing outputs to timestamped backups before
starting a clean file.  Fallback responses are retried a bounded number of
times.  A second CSV aggregates judge correctness accuracy (threshold 0.5),
recovery/degradation counts, faithfulness, and token F1.

The backend must already be running.  Start with a smoke test, for example::

    python -m backend.benchmark_repair_effectiveness --questions 1

Do not use ``synthetic-evaluator`` as either the generation model or judge.
Any generation or judge fallback is retained and flagged in the detail CSV,
but excluded from primary and secondary aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime


MATCHED_REPAIRS = (
    ("spelling", "fix_spelling"),
    ("conflict", "contradiction_filter"),
    ("conflict", "contradiction_filter_text_only"),
    ("shuffle", "fix_shuffle"),
    ("truncate", "fix_truncate"),
    ("alter_facts", "fix_alter_facts"),
    ("alter_facts", "fix_alter_facts_drop"),
)
CORRECTNESS_THRESHOLD = 0.5
DEFAULT_DETAILS = pathlib.Path("results/repair_effectiveness_per_question.csv")
DEFAULT_SUMMARY = pathlib.Path("results/repair_effectiveness_summary.csv")

DETAIL_FIELDS = (
    "question_id", "question", "perturbation", "repair", "seed",
    "generation_model", "judge_model", "embedding_provider", "search_type",
    "top_k", "typo_rate", "perturbation_applied_off",
    "perturbation_applied_on", "repair_applied", "repair_status",
    "repair_summary", "valid_for_analysis",
    "exclusion_reason", "correctness_off", "correctness_on", "correct_off",
    "correct_on", "correctness_delta", "recovered", "degraded",
    "faithfulness_off", "faithfulness_on", "faithfulness_delta", "f1_off",
    "f1_on", "f1_delta", "judge_source_off", "judge_source_on",
    "generation_source_off", "generation_source_on", "warning_off",
    "warning_on", "answer_off", "answer_on", "ground_truth",
)


def http_json(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _variant(response, perturbation):
    variants = response.get("variants") or []
    for variant in variants:
        if variant.get("methods") == [perturbation]:
            return variant
    if len(variants) == 1:
        return variants[0]
    raise ValueError(f"Backend response did not contain a {perturbation!r} variant")


def _change_applied(variant, method):
    return any(
        change.get("method") == method and change.get("status") == "applied"
        for change in variant.get("changes") or []
    )


def _change(variant, method):
    return next(
        (change for change in variant.get("changes") or []
         if change.get("method") == method),
        None,
    )


def _synthetic_reason(variant):
    metrics = variant.get("metrics") or {}
    judge_source = str(metrics.get("judge_model") or "")
    generation_source = str(variant.get("perturbed_generation_source") or "")
    warning = str(variant.get("warning") or "")
    reasons = []
    if generation_source.casefold() == "synthetic" or "synthetic" in warning.casefold():
        reasons.append("synthetic generation fallback")
    if judge_source.casefold() == "synthetic-evaluator":
        reasons.append("synthetic judge fallback")
    if metrics.get("judge_correctness_perturbed") is None:
        reasons.append("missing judge correctness")
    return "; ".join(dict.fromkeys(reasons))


def make_detail_row(question, perturbation, repair, seed, off_response, on_response, args):
    off = _variant(off_response, perturbation)
    on = _variant(on_response, perturbation)
    off_metrics, on_metrics = off.get("metrics") or {}, on.get("metrics") or {}
    correctness_off = _as_float(off_metrics.get("judge_correctness_perturbed"))
    correctness_on = _as_float(on_metrics.get("judge_correctness_perturbed"))
    faithfulness_off = _as_float(off_metrics.get("judge_faithfulness_perturbed"))
    faithfulness_on = _as_float(on_metrics.get("judge_faithfulness_perturbed"))
    f1_off = _as_float(off_metrics.get("answer_f1_perturbed"))
    f1_on = _as_float(on_metrics.get("answer_f1_perturbed"))
    repair_change = _change(on, repair)
    exclusion_parts = [_synthetic_reason(off), _synthetic_reason(on)]
    if repair_change is None:
        exclusion_parts.append("missing repair result")
    exclusion = "; ".join(dict.fromkeys(filter(None, exclusion_parts)))
    valid = not exclusion
    correct_off = valid and correctness_off >= CORRECTNESS_THRESHOLD
    correct_on = valid and correctness_on >= CORRECTNESS_THRESHOLD

    def delta(after, before):
        return after - before if valid and after is not None and before is not None else None

    return {
        "question_id": str(question["id"]), "question": question["question"],
        "perturbation": perturbation, "repair": repair, "seed": seed,
        "generation_model": args.llm_model, "judge_model": args.judge_model,
        "embedding_provider": args.embedding_provider, "search_type": args.search_type,
        "top_k": args.top_k, "typo_rate": args.typo_rate,
        "perturbation_applied_off": _change_applied(off, perturbation),
        "perturbation_applied_on": _change_applied(on, perturbation),
        "repair_applied": _change_applied(on, repair),
        "repair_status": repair_change.get("status") if repair_change else "missing",
        "repair_summary": repair_change.get("summary") if repair_change else "",
        "valid_for_analysis": valid, "exclusion_reason": exclusion,
        "correctness_off": correctness_off, "correctness_on": correctness_on,
        "correct_off": correct_off if valid else "", "correct_on": correct_on if valid else "",
        "correctness_delta": delta(correctness_on, correctness_off),
        "recovered": bool(valid and not correct_off and correct_on),
        "degraded": bool(valid and correct_off and not correct_on),
        "faithfulness_off": faithfulness_off, "faithfulness_on": faithfulness_on,
        "faithfulness_delta": delta(faithfulness_on, faithfulness_off),
        "f1_off": f1_off, "f1_on": f1_on, "f1_delta": delta(f1_on, f1_off),
        "judge_source_off": off_metrics.get("judge_model"),
        "judge_source_on": on_metrics.get("judge_model"),
        "generation_source_off": off.get("perturbed_generation_source"),
        "generation_source_on": on.get("perturbed_generation_source"),
        "warning_off": off.get("warning"), "warning_on": on.get("warning"),
        "answer_off": off.get("perturbed_answer"), "answer_on": on.get("perturbed_answer"),
        "ground_truth": on.get("ground_truth") or question.get("ground_truth"),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["perturbation"], row["repair"])].append(row)
    output = []
    for (perturbation, repair), group in sorted(groups.items()):
        valid = [r for r in group if str(r.get("valid_for_analysis", "")).lower() == "true"]

        def values(field):
            return [v for r in valid if (v := _as_float(r.get(field))) is not None]

        def mean(field):
            vals = values(field)
            return sum(vals) / len(vals) if vals else None

        off_correct = sum(str(r.get("correct_off", "")).lower() == "true" for r in valid)
        on_correct = sum(str(r.get("correct_on", "")).lower() == "true" for r in valid)
        n = len(valid)
        output.append({
            "perturbation": perturbation, "repair": repair,
            "rows_total": len(group), "rows_valid": n, "rows_excluded": len(group) - n,
            "accuracy_off": off_correct / n if n else None,
            "accuracy_on": on_correct / n if n else None,
            "accuracy_delta": (on_correct - off_correct) / n if n else None,
            "recovered_count": sum(str(r.get("recovered", "")).lower() == "true" for r in valid),
            "degraded_count": sum(str(r.get("degraded", "")).lower() == "true" for r in valid),
            "mean_correctness_off": mean("correctness_off"),
            "mean_correctness_on": mean("correctness_on"),
            "mean_correctness_delta": mean("correctness_delta"),
            "mean_faithfulness_off": mean("faithfulness_off"),
            "mean_faithfulness_on": mean("faithfulness_on"),
            "mean_faithfulness_delta": mean("faithfulness_delta"),
            "mean_f1_off": mean("f1_off"), "mean_f1_on": mean("f1_on"),
            "mean_f1_delta": mean("f1_delta"),
        })
    return output


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_fields(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle), [])


def _append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_detail_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _backup_outputs(paths):
    """Move prior outputs aside so --no-resume always starts a clean run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = []
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        candidate = path.with_name(f"{path.name}.bak-{stamp}")
        suffix = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.name}.bak-{stamp}-{suffix}")
            suffix += 1
        path.replace(candidate)
        backups.append((path, candidate))
    return backups


def _upgrade_detail_schema(path, rows):
    """Safely add newly introduced columns before appending resumed rows."""
    if not path.exists() or _read_fields(path) == list(DETAIL_FIELDS):
        return None
    backup = _backup_outputs((path,))[0][1]
    for row in rows:
        if not row.get("repair_status"):
            applied = str(row.get("repair_applied", "")).casefold() == "true"
            row["repair_status"] = "applied" if applied else "legacy_unknown"
        row.setdefault("repair_summary", "")
    _write_detail_rows(path, rows)
    return backup


def _response_retry_reason(response):
    if not response.get("variants"):
        return "missing variants"
    reasons = []
    for variant in response.get("variants") or []:
        reason = _synthetic_reason(variant)
        if reason:
            reasons.append(reason)
    return "; ".join(dict.fromkeys(reasons))


def _request_with_retries(url, payload, timeout, max_attempts, retry_delay):
    """Retry transient request failures and fallback responses, keeping the last result."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = http_json(url, payload, timeout)
        except (urllib.error.URLError, OSError) as exc:
            if attempt == max_attempts:
                raise
            print(f"  attempt {attempt}/{max_attempts} failed: {exc}; retrying", flush=True)
        else:
            reason = _response_retry_reason(response)
            if not reason or attempt == max_attempts:
                return response
            print(f"  attempt {attempt}/{max_attempts} excluded ({reason}); retrying", flush=True)
        if retry_delay:
            time.sleep(retry_delay)
    raise AssertionError("retry loop exhausted without returning or raising")


def _write_summary(path, rows):
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0]) if summary else ["perturbation", "repair"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _payload(question, perturbation, repair, seed, args):
    return {
        "question_id": question["id"], "baseline_query": question["question"],
        "ground_truth": question["ground_truth"], "contexts": question["contexts"],
        "gold_titles": question.get("gold_titles") or [],
        "supporting_titles": question.get("supporting_titles"),
        "perturbation_types": [perturbation], "mode": "separate", "seed": seed,
        "repair_methods": [repair] if repair else None, "typo_rate": args.typo_rate,
        "embedding_provider": args.embedding_provider, "search_type": args.search_type,
        "top_k": args.top_k, "llm_model": args.llm_model,
        "use_llm_judge": True, "judge_model": args.judge_model,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=int, default=1,
                        help="number of questions (smoke default: 1; full dataset: 80)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000,
                        help="question i uses seed + dataset index")
    parser.add_argument("--llm-model", default="ollama/qwen3.5:9b")
    parser.add_argument("--judge-model", default="ollama/qwen3.5:9b")
    parser.add_argument("--embedding-provider", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--search-type", choices=("similarity", "mmr"), default="similarity")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--typo-rate", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="maximum attempts for each backend evaluation (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                        help="seconds between retry attempts (default: 1)")
    parser.add_argument("--details-output", type=pathlib.Path, default=DEFAULT_DETAILS)
    parser.add_argument("--summary-output", type=pathlib.Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--no-resume", action="store_true",
                        help="start clean and move existing output CSVs to timestamped backups")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.llm_model == "synthetic-evaluator" or args.judge_model == "synthetic-evaluator":
        parser.error("synthetic-evaluator is not valid for the generation model or primary judge")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.retry_delay < 0:
        parser.error("--retry-delay cannot be negative")
    backend = args.backend.rstrip("/")
    try:
        dataset = http_json(f"{backend}/api/dataset", timeout=args.timeout)["dataset"]
        registered = {item["name"] for item in http_json(f"{backend}/api/repairs", timeout=args.timeout)["repairs"]}
    except (urllib.error.URLError, OSError, KeyError) as exc:
        parser.error(f"cannot reach a compatible backend at {backend}: {exc}")
    missing = sorted({repair for _, repair in MATCHED_REPAIRS} - registered)
    if missing:
        parser.error("backend is missing required repairs: " + ", ".join(missing))
    questions = dataset[args.offset:args.offset + args.questions]
    existing = [] if args.no_resume else _read_rows(args.details_output)
    done = {(str(r["question_id"]), r["perturbation"], r["repair"], str(r["seed"])) for r in existing}
    plan = [
        (question, perturbation, repair, args.seed + args.offset + index)
        for index, question in enumerate(questions)
        for perturbation, repair in MATCHED_REPAIRS
        if (str(question["id"]), perturbation, repair,
            str(args.seed + args.offset + index)) not in done
    ]
    print(f"Questions: {len(questions)} | repair pairs: {len(MATCHED_REPAIRS)} | "
          f"comparisons remaining: {len(plan)} | backend requests: {2 * len(plan)}")
    if args.dry_run:
        return 0
    if existing:
        schema_backup = _upgrade_detail_schema(args.details_output, existing)
        if schema_backup:
            print(f"Upgraded detail CSV schema; backup saved to {schema_backup}", flush=True)
    if args.no_resume:
        backups = _backup_outputs((args.details_output, args.summary_output))
        for original, backup in backups:
            print(f"Backed up {original} to {backup}", flush=True)
    all_rows = list(existing)
    for number, (question, perturbation, repair, seed) in enumerate(plan, 1):
        print(f"[{number}/{len(plan)}] {question['id']} {perturbation} -> {repair}", flush=True)
        off = _request_with_retries(
            f"{backend}/api/run-evaluation",
            _payload(question, perturbation, None, seed, args), args.timeout,
            args.max_attempts, args.retry_delay,
        )
        on = _request_with_retries(
            f"{backend}/api/run-evaluation",
            _payload(question, perturbation, repair, seed, args), args.timeout,
            args.max_attempts, args.retry_delay,
        )
        row = make_detail_row(question, perturbation, repair, seed, off, on, args)
        _append_row(args.details_output, row)
        all_rows.append(row)
        _write_summary(args.summary_output, all_rows)
    _write_summary(args.summary_output, all_rows)
    excluded = sum(str(r.get("valid_for_analysis", "")).lower() != "true" for r in all_rows)
    print(f"Wrote {args.details_output} and {args.summary_output}; "
          f"{excluded}/{len(all_rows)} rows excluded/flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
