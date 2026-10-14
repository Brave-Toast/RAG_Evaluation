"""
Modular LLM-as-a-Judge evaluator and FastAPI integration hook for RAG Triad metrics:
- Answer Correctness (Generated Answer vs. Ground Truth)
- Context Faithfulness (Generated Answer vs. Supplied Context)
- Context Relevance (Supplied Context vs. Target Question)
"""

import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

try:
    from backend.isolation_metrics import calculate_token_f1, diagnose_failure, normalize_text
except ImportError:
    try:
        from .isolation_metrics import calculate_token_f1, diagnose_failure, normalize_text
    except ImportError:
        from isolation_metrics import calculate_token_f1, diagnose_failure, normalize_text

SUPPORTED_JUDGES = (
    "ollama/qwen3.5:9b",
    "ollama/llama3",
    "ollama/mistral",
    "gemini-3-flash-preview",
    "synthetic-evaluator",
)
DEFAULT_JUDGE_MODEL = "ollama/qwen3.5:9b"

JUDGE_SYSTEM_PROMPT = """You are an impartial, rigorous RAG Evaluation Judge.
Evaluate the Generated Answer against the Question, Ground Truth Answer, and Supplied Context Passages.

Score the following three dimensions on a continuous scale from 0.0 to 1.0:
1. "correctness": Does the Generated Answer accurately convey the same factual answer as the Ground Truth Answer? Do NOT penalize full-sentence phrasing if the core entity/fact matches the Ground Truth.
2. "faithfulness": Is every claim in the Generated Answer strictly supported by the Supplied Context Passages without ungrounded extrapolation or hallucination? (If the context contains conflicting claims and the answer explicitly notes the conflict, score faithfulness high).
3. "context_relevance": Do the Supplied Context Passages contain the necessary evidence to answer the Question?

Assign a "verdict" from one of:
- "PASS": correctness >= 0.5 and faithfulness >= 0.5
- "HALLUCINATION": correctness < 0.5 and faithfulness < 0.5
- "UNFAITHFUL_LEAKAGE": correctness >= 0.5 and faithfulness < 0.5 (answered from parametric memory, not context)
- "CONTEXT_STARVED": correctness < 0.5 and faithfulness >= 0.5 (faithfully abstained or failed because context lacked evidence)

Respond ONLY with a valid JSON object matching this exact schema:
{
  "correctness": <float 0.0 to 1.0>,
  "faithfulness": <float 0.0 to 1.0>,
  "context_relevance": <float 0.0 to 1.0>,
  "verdict": "PASS" | "HALLUCINATION" | "UNFAITHFUL_LEAKAGE" | "CONTEXT_STARVED",
  "rationale": "<concise 1-2 sentence explanation>"
}"""


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return round(default, 4)


def _derive_verdict(correctness: float, faithfulness: float) -> str:
    if correctness >= 0.5 and faithfulness >= 0.5:
        return "PASS"
    if correctness >= 0.5 and faithfulness < 0.5:
        return "UNFAITHFUL_LEAKAGE"
    if correctness < 0.5 and faithfulness >= 0.5:
        return "CONTEXT_STARVED"
    return "HALLUCINATION"


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    for candidate in (cleaned, raw_text.strip()):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    match = re.search(r"\{[\s\S]*\}", cleaned) or re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def _normalize_context_list(contexts: List[Any]) -> List[Dict[str, str]]:
    normalized = []
    for doc in contexts or []:
        if isinstance(doc, dict):
            normalized.append({
                "title": str(doc.get("title", "")),
                "text": str(doc.get("text", "")),
            })
        else:
            normalized.append({
                "title": str(getattr(doc, "title", "")),
                "text": str(getattr(doc, "text", "")),
            })
    return normalized


def evaluate_synthetic_judge(
    query: str,
    contexts: List[Any],
    generated_answer: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Deterministic heuristic judge used when judge_model == 'synthetic-evaluator'
    or as a resilient fallback when local/cloud LLM endpoints are unreachable.
    """
    docs = _normalize_context_list(contexts)
    norm_ans = normalize_text(generated_answer)
    norm_gt = normalize_text(ground_truth)
    norm_ctx = normalize_text(" ".join(f"{d['title']} {d['text']}" for d in docs))
    norm_q = normalize_text(query)

    f1 = calculate_token_f1(generated_answer, ground_truth)

    negated = any(
        phrase in norm_ans
        for phrase in (
            "conflicting sources refute",
            "does not state",
            "not mentioned",
            "cannot be determined",
        )
    )
    if norm_gt and norm_gt in norm_ans and not negated:
        correctness = max(f1, 0.95)
    elif negated and norm_gt and norm_gt in norm_ans:
        correctness = min(f1, 0.25)
    else:
        correctness = f1

    q_tokens = [t for t in norm_q.split() if len(t) > 2]
    q_overlap = (
        sum(1 for t in q_tokens if t in norm_ctx) / len(q_tokens)
        if q_tokens else 1.0
    )
    gt_in_ctx = 1.0 if (norm_gt and norm_gt in norm_ctx) else 0.0
    context_relevance = _clamp_score(0.6 * gt_in_ctx + 0.4 * q_overlap)

    ans_tokens = [t for t in norm_ans.split() if len(t) > 2]
    if not docs:
        faithfulness = 0.0 if ans_tokens else 1.0
    elif not ans_tokens:
        faithfulness = 0.0
    else:
        grounded_count = sum(1 for t in ans_tokens if t in norm_ctx)
        faithfulness = _clamp_score(grounded_count / len(ans_tokens))

    verdict = _derive_verdict(correctness, faithfulness)
    rationale = (
        f"Synthetic judge evaluated semantic containment and token grounding "
        f"(Correctness={correctness:.2f}, Faithfulness={faithfulness:.2f}, Relevance={context_relevance:.2f})."
    )
    return {
        "correctness": _clamp_score(correctness),
        "faithfulness": _clamp_score(faithfulness),
        "context_relevance": _clamp_score(context_relevance),
        "verdict": verdict,
        "rationale": rationale,
        "source": "synthetic-evaluator",
        "warning": None,
    }


def _build_judge_user_prompt(
    query: str,
    contexts: List[Dict[str, str]],
    generated_answer: str,
    ground_truth: str,
) -> str:
    if contexts:
        formatted_ctx = "\n".join(
            f"[{i + 1}] [{c['title']}]: {c['text']}" for i, c in enumerate(contexts)
        )
    else:
        formatted_ctx = "(No context passages supplied)"

    return (
        f"Question: {query}\n\n"
        f"Ground Truth Answer: {ground_truth}\n\n"
        f"Supplied Context Passages:\n{formatted_ctx}\n\n"
        f"Generated Answer: {generated_answer}\n\n"
        "Return your evaluation strictly as JSON."
    )


async def evaluate_with_llm_judge(
    query: str,
    contexts: List[Any],
    generated_answer: str,
    ground_truth: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    gemini_api_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Runs LLM-as-a-Judge and returns (normalized_judge_dict, warning_message).
    """
    docs = _normalize_context_list(contexts)
    if not judge_model or judge_model == "synthetic-evaluator":
        res = evaluate_synthetic_judge(query, docs, generated_answer, ground_truth)
        return res, None

    user_prompt = _build_judge_user_prompt(query, docs, generated_answer, ground_truth)
    raw_output = ""

    try:
        if judge_model.startswith("ollama/"):
            model_tag = judge_model.split("ollama/", 1)[1]
            payload = {
                "model": model_tag,
                "system": JUDGE_SYSTEM_PROMPT,
                "prompt": user_prompt,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512,
                },
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post("http://localhost:11434/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_output = (data.get("response") or "").strip()
                if not raw_output and data.get("thinking"):
                    raw_output = str(data.get("thinking")).strip()

        elif judge_model.startswith("gemini"):
            api_key = (
                gemini_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY for Gemini judge evaluation.")
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{judge_model}:generateContent?key={api_key}"
            )
            payload = {
                "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            }
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts") or []
                    if parts:
                        raw_output = (parts[0].get("text") or "").strip()
        else:
            raise ValueError(f"Unsupported judge model provider: {judge_model}")

        parsed = _extract_json_object(raw_output)
        if not parsed:
            raise ValueError(
                f"Judge model '{judge_model}' did not return valid JSON: {raw_output[:200]}"
            )

        correctness = _clamp_score(parsed.get("correctness"), 0.0)
        faithfulness = _clamp_score(parsed.get("faithfulness"), 0.0)
        context_relevance = _clamp_score(parsed.get("context_relevance"), 0.0)
        verdict = str(parsed.get("verdict") or "").upper()
        if verdict not in {"PASS", "HALLUCINATION", "UNFAITHFUL_LEAKAGE", "CONTEXT_STARVED"}:
            verdict = _derive_verdict(correctness, faithfulness)
        rationale = str(parsed.get("rationale") or "Evaluated by LLM Judge.").strip()

        return {
            "correctness": correctness,
            "faithfulness": faithfulness,
            "context_relevance": context_relevance,
            "verdict": verdict,
            "rationale": rationale,
            "source": judge_model,
            "warning": None,
        }, None

    except Exception as exc:
        warning = f"LLM Judge ({judge_model}) fell back to synthetic judge: {exc}"
        fallback = evaluate_synthetic_judge(query, docs, generated_answer, ground_truth)
        fallback["warning"] = warning
        return fallback, warning


def _get_backend_module():
    return sys.modules.get("backend.app") or sys.modules.get("app")


def _resolve_gold_titles(req_data: Dict[str, Any]) -> List[str]:
    gold = req_data.get("gold_titles")
    if isinstance(gold, list) and gold:
        return [str(t) for t in gold]
    qid = req_data.get("question_id")
    backend_mod = _get_backend_module()
    if qid and backend_mod and hasattr(backend_mod, "load_hotpotqa_dataset"):
        try:
            dataset = backend_mod.load_hotpotqa_dataset() or []
            for item in dataset:
                if isinstance(item, dict) and item.get("id") == qid:
                    return [str(t) for t in (item.get("gold_titles") or [])]
        except Exception:
            pass
    return []


def _apply_judge_to_metrics(
    metrics: Dict[str, Any],
    use_llm_judge: bool,
    baseline_judge: Optional[Dict[str, Any]],
    perturbed_judge: Optional[Dict[str, Any]],
    perturbed_contexts: List[Any],
    gold_titles: List[str],
    perturbed_answer: str,
    ground_truth: str,
) -> None:
    metrics["use_llm_judge"] = bool(use_llm_judge)
    metrics["judge_model"] = perturbed_judge["source"] if perturbed_judge else None
    metrics["judge_correctness_baseline"] = (
        baseline_judge["correctness"] if baseline_judge else None
    )
    metrics["judge_correctness_perturbed"] = (
        perturbed_judge["correctness"] if perturbed_judge else None
    )
    metrics["judge_faithfulness_baseline"] = (
        baseline_judge["faithfulness"] if baseline_judge else None
    )
    metrics["judge_faithfulness_perturbed"] = (
        perturbed_judge["faithfulness"] if perturbed_judge else None
    )
    metrics["judge_context_relevance_baseline"] = (
        baseline_judge["context_relevance"] if baseline_judge else None
    )
    metrics["judge_context_relevance_perturbed"] = (
        perturbed_judge["context_relevance"] if perturbed_judge else None
    )
    metrics["judge_verdict_baseline"] = (
        baseline_judge["verdict"] if baseline_judge else None
    )
    metrics["judge_verdict_perturbed"] = (
        perturbed_judge["verdict"] if perturbed_judge else None
    )
    metrics["judge_rationale_baseline"] = (
        baseline_judge["rationale"] if baseline_judge else None
    )
    metrics["judge_rationale_perturbed"] = (
        perturbed_judge["rationale"] if perturbed_judge else None
    )

    if use_llm_judge and perturbed_judge:
        titles = [
            (c.get("title") if isinstance(c, dict) else getattr(c, "title", ""))
            for c in (perturbed_contexts or [])
        ]
        diag = diagnose_failure(
            retrieved_titles=titles,
            gold_titles=gold_titles,
            generated_answer=perturbed_answer,
            ground_truth=ground_truth,
            judge_result=perturbed_judge,
        )
        metrics["failure_attribution"] = diag["verdict"]
        metrics["failure_root_cause"] = diag["root_cause"]


def _update_stage_logs_with_judge(
    evaluation_id: Optional[str],
    use_llm_judge: bool,
    requested_model: Optional[str],
    baseline_judge: Optional[Dict[str, Any]],
    variant_judges: List[Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]],
) -> None:
    backend_mod = _get_backend_module()
    log_path = getattr(backend_mod, "STAGE_LOG_PATH", None) if backend_mod else None
    if not log_path:
        return
    path_obj = pathlib.Path(log_path)
    if not path_obj.exists():
        return
    try:
        lines = path_obj.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        matching_indices = [
            i for i, r in enumerate(records)
            if evaluation_id and r.get("evaluation_id") == evaluation_id
        ]
        if not matching_indices and records:
            matching_indices = list(
                range(max(0, len(records) - len(variant_judges)), len(records))
            )
        for idx, (p_judge, attr, root_cause) in zip(matching_indices, variant_judges):
            records[idx]["judge_stage"] = {
                "enabled": bool(use_llm_judge),
                "requested_model": requested_model if use_llm_judge else None,
                "baseline": baseline_judge,
                "perturbed": p_judge,
            }
            if use_llm_judge and attr and root_cause:
                records[idx]["failure_attribution"] = attr
                records[idx]["failure_root_cause"] = root_cause
        path_obj.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
    except Exception:
        pass


async def enrich_evaluation_result_with_judge(
    result: Dict[str, Any],
    req_data: Dict[str, Any],
    use_llm_judge: bool,
    judge_model: str,
) -> Dict[str, Any]:
    """Enriches a completed /api/run-evaluation JSON payload with LLM Judge scores."""
    ground_truth = result.get("ground_truth") or req_data.get("ground_truth") or ""
    gemini_api_key = req_data.get("gemini_api_key")
    gold_titles = _resolve_gold_titles(req_data)

    baseline_judge = None
    warnings: List[str] = []

    if use_llm_judge:
        baseline_judge, b_warn = await evaluate_with_llm_judge(
            query=result.get("baseline_query") or req_data.get("baseline_query") or "",
            contexts=result.get("baseline_contexts") or [],
            generated_answer=result.get("baseline_answer") or "",
            ground_truth=ground_truth,
            judge_model=judge_model,
            gemini_api_key=gemini_api_key,
        )
        if b_warn:
            warnings.append(b_warn)

    variants = result.get("variants") or []
    variant_log_updates: List[Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]] = []

    for variant in variants:
        p_judge = None
        if use_llm_judge:
            p_judge, p_warn = await evaluate_with_llm_judge(
                query=variant.get("perturbed_query") or result.get("perturbed_query") or "",
                contexts=variant.get("perturbed_contexts") or [],
                generated_answer=variant.get("perturbed_answer") or "",
                ground_truth=ground_truth,
                judge_model=judge_model,
                gemini_api_key=gemini_api_key,
            )
            if p_warn:
                warnings.append(p_warn)
                variant["warning"] = (
                    f"{variant['warning']} | {p_warn}" if variant.get("warning") else p_warn
                )

        v_metrics = variant.setdefault("metrics", {})
        _apply_judge_to_metrics(
            metrics=v_metrics,
            use_llm_judge=use_llm_judge,
            baseline_judge=baseline_judge,
            perturbed_judge=p_judge,
            perturbed_contexts=variant.get("perturbed_contexts") or [],
            gold_titles=gold_titles,
            perturbed_answer=variant.get("perturbed_answer") or "",
            ground_truth=ground_truth,
        )
        variant_log_updates.append(
            (p_judge, v_metrics.get("failure_attribution"), v_metrics.get("failure_root_cause"))
        )

    if variants and isinstance(result.get("metrics"), dict):
        first_v_metrics = variants[0].get("metrics") or {}
        for k, v in first_v_metrics.items():
            if k.startswith("judge_") or k in ("use_llm_judge", "failure_attribution", "failure_root_cause"):
                result["metrics"][k] = v
    elif isinstance(result.get("metrics"), dict):
        p_judge = None
        if use_llm_judge:
            p_judge, p_warn = await evaluate_with_llm_judge(
                query=result.get("perturbed_query") or "",
                contexts=result.get("perturbed_contexts") or [],
                generated_answer=result.get("perturbed_answer") or "",
                ground_truth=ground_truth,
                judge_model=judge_model,
                gemini_api_key=gemini_api_key,
            )
            if p_warn:
                warnings.append(p_warn)
        _apply_judge_to_metrics(
            metrics=result["metrics"],
            use_llm_judge=use_llm_judge,
            baseline_judge=baseline_judge,
            perturbed_judge=p_judge,
            perturbed_contexts=result.get("perturbed_contexts") or [],
            gold_titles=gold_titles,
            perturbed_answer=result.get("perturbed_answer") or "",
            ground_truth=ground_truth,
        )
        variant_log_updates.append(
            (
                p_judge,
                result["metrics"].get("failure_attribution"),
                result["metrics"].get("failure_root_cause"),
            )
        )

    if warnings:
        joined = " | ".join(dict.fromkeys(warnings))
        result["warning"] = f"{result['warning']} | {joined}" if result.get("warning") else joined

    _update_stage_logs_with_judge(
        evaluation_id=result.get("evaluation_id"),
        use_llm_judge=use_llm_judge,
        requested_model=judge_model,
        baseline_judge=baseline_judge,
        variant_judges=variant_log_updates,
    )
    return result


def register_llm_judge(app: FastAPI) -> None:
    """
    Registers /api/judges and attaches the LLM-as-a-Judge middleware to the FastAPI app.
    Safe to call multiple times (idempotent).
    """
    if getattr(app.state, "llm_judge_registered", False):
        return
    app.state.llm_judge_registered = True

    @app.get("/api/judges")
    async def list_judges():
        return {
            "default": DEFAULT_JUDGE_MODEL,
            "judges": [
                {"name": "ollama/qwen3.5:9b", "label": "Ollama / qwen3.5:9b (Local Judge)"},
                {"name": "ollama/llama3", "label": "Ollama / llama3 (Local Judge)"},
                {"name": "ollama/mistral", "label": "Ollama / mistral (Local Judge)"},
                {"name": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Cloud Judge)"},
                {"name": "synthetic-evaluator", "label": "Synthetic Heuristic Judge (Offline)"},
            ],
        }

    @app.middleware("http")
    async def llm_judge_middleware(request: Request, call_next):
        path = request.url.path
        if request.method != "POST" or path not in (
            "/api/run-evaluation",
            "/api/run-evaluation/stream",
        ):
            return await call_next(request)

        raw_body = await request.body()
        try:
            req_data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            req_data = {}

        use_llm_judge = bool(req_data.get("use_llm_judge", False))
        judge_model = str(req_data.get("judge_model") or DEFAULT_JUDGE_MODEL)

        if use_llm_judge and judge_model not in SUPPORTED_JUDGES:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": (
                        f"Unsupported judge_model '{judge_model}'. "
                        f"Supported: {list(SUPPORTED_JUDGES)}"
                    )
                },
            )

        # Strip judge-specific keys before forwarding to EvaluationRequest in case extra='forbid'
        forwarded_data = {
            k: v for k, v in req_data.items() if k not in ("use_llm_judge", "judge_model")
        }
        forwarded_bytes = json.dumps(forwarded_data).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": forwarded_bytes, "more_body": False}

        forwarded_req = Request(request.scope, receive)
        response = await call_next(forwarded_req)

        if response.status_code != 200:
            return response

        if path == "/api/run-evaluation":
            resp_bytes = b""
            async for chunk in response.body_iterator:
                resp_bytes += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            try:
                result_obj = json.loads(resp_bytes.decode("utf-8"))
            except Exception:
                return Response(
                    content=resp_bytes,
                    status_code=response.status_code,
                    media_type=response.media_type,
                )

            enriched = await enrich_evaluation_result_with_judge(
                result=result_obj,
                req_data=req_data,
                use_llm_judge=use_llm_judge,
                judge_model=judge_model,
            )
            return JSONResponse(content=enriched, status_code=200)

        # Streaming endpoint: /api/run-evaluation/stream
        async def judge_stream_wrapper():
            buffer = ""
            decoder_chunks = response.body_iterator
            async for chunk in decoder_chunks:
                text_chunk = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                buffer += text_chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        event_obj = json.loads(line)
                    except Exception:
                        yield (line + "\n").encode("utf-8")
                        continue

                    if event_obj.get("event") == "complete" and isinstance(event_obj.get("result"), dict):
                        if use_llm_judge:
                            progress_msg = {
                                "event": "progress",
                                "message": f"Running LLM Judge ({judge_model})...",
                            }
                            yield (json.dumps(progress_msg) + "\n").encode("utf-8")
                        event_obj["result"] = await enrich_evaluation_result_with_judge(
                            result=event_obj["result"],
                            req_data=req_data,
                            use_llm_judge=use_llm_judge,
                            judge_model=judge_model,
                        )
                    yield (json.dumps(event_obj) + "\n").encode("utf-8")

            if buffer.strip():
                try:
                    event_obj = json.loads(buffer)
                    if event_obj.get("event") == "complete" and isinstance(event_obj.get("result"), dict):
                        event_obj["result"] = await enrich_evaluation_result_with_judge(
                            result=event_obj["result"],
                            req_data=req_data,
                            use_llm_judge=use_llm_judge,
                            judge_model=judge_model,
                        )
                    yield (json.dumps(event_obj) + "\n").encode("utf-8")
                except Exception:
                    yield (buffer + "\n").encode("utf-8")

        return StreamingResponse(
            judge_stream_wrapper(),
            status_code=response.status_code,
            media_type=response.media_type or "application/x-ndjson",
        )