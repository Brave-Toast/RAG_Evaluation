"""
Stage Isolation Metrics and Root-Cause Failure Attribution for RAG Pipelines.
"""

from collections import Counter
import re


def normalize_text(text: str | None) -> str:
    """Lowercases, strips punctuation, and normalizes whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", "", str(text).lower())
    return " ".join(cleaned.split())


def calculate_token_f1(prediction: str | None, ground_truth: str | None) -> float:
    """Computes token-level F1 overlap between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def calculate_exact_match(prediction: str | None, ground_truth: str | None) -> float:
    """Returns 1.0 if normalized prediction exactly matches normalized ground truth, else 0.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def calculate_context_recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Fraction of gold supporting document titles present in retrieved_titles."""
    if not gold_titles:
        return 1.0
    if not retrieved_titles:
        return 0.0

    ret_set = {str(t).strip().lower() for t in retrieved_titles if t}
    gold_set = {str(t).strip().lower() for t in gold_titles if t}
    if not gold_set:
        return 1.0

    hits = len(ret_set & gold_set)
    return round(hits / len(gold_set), 4)


def calculate_context_precision(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Average Precision (AP) of gold titles within the ranked retrieved_titles list."""
    if not gold_titles:
        return 1.0
    if not retrieved_titles:
        return 0.0

    gold_set = {str(t).strip().lower() for t in gold_titles if t}
    if not gold_set:
        return 1.0

    hits = 0
    precision_sum = 0.0
    seen = set()

    for rank, title in enumerate(retrieved_titles, start=1):
        norm_title = str(title or "").strip().lower()
        if norm_title in gold_set and norm_title not in seen:
            seen.add(norm_title)
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return round(precision_sum / hits, 4)


def diagnose_failure(
    retrieved_titles: list[str],
    gold_titles: list[str],
    generated_answer: str,
    ground_truth: str,
    f1_threshold: float = 0.5,
    judge_result: dict | None = None,
) -> dict:
    """
    Classifies a RAG execution into one of the 4 diagnostic attribution quadrants:
      1. SUCCESS:            Sufficient context retrieved AND accurate answer generated.
      2. RETRIEVAL_FAILURE:  Insufficient context retrieved -> caused inaccurate answer.
      3. GENERATION_FAILURE: Sufficient context retrieved, but generator hallucinated / failed.
      4. PARAMETRIC_LEAKAGE: Insufficient context retrieved, yet generator answered accurately.

    When `judge_result` is provided (LLM-as-a-Judge enabled), answer accuracy is evaluated
    using `judge_result["correctness"] >= f1_threshold`.
    """
    context_recall = calculate_context_recall(retrieved_titles, gold_titles)
    context_precision = calculate_context_precision(retrieved_titles, gold_titles)
    answer_f1 = calculate_token_f1(generated_answer, ground_truth)
    exact_match = calculate_exact_match(generated_answer, ground_truth)

    recall_threshold = 1.0 if len(gold_titles) <= 1 else 0.5
    retrieval_ok = context_recall >= recall_threshold

    if judge_result and judge_result.get("correctness") is not None:
        judge_corr = float(judge_result["correctness"])
        judge_faith = float(judge_result.get("faithfulness", 1.0))
        judge_rationale = str(judge_result.get("rationale") or "").strip()
        answer_ok = judge_corr >= f1_threshold
        score_summary = (
            f"Judge Correctness: {judge_corr:.2f}, Faithfulness: {judge_faith:.2f}, "
            f"Token F1: {answer_f1:.2f}"
        )
    else:
        answer_ok = answer_f1 >= f1_threshold
        judge_rationale = ""
        score_summary = f"Token F1: {answer_f1:.2f}"

    rationale_suffix = f" [Judge: {judge_rationale}]" if judge_rationale else ""

    if retrieval_ok and answer_ok:
        verdict = "SUCCESS"
        root_cause = (
            f"Pipeline succeeded: Retriever supplied necessary context "
            f"(Context Recall: {context_recall:.2f}, Precision: {context_precision:.2f}) "
            f"and LLM answered accurately ({score_summary}).{rationale_suffix}"
        )
    elif not retrieval_ok and not answer_ok:
        verdict = "RETRIEVAL_FAILURE"
        root_cause = (
            f"Cascading Retrieval Failure: Retriever missed gold evidence "
            f"(Context Recall: {context_recall:.2f} < {recall_threshold:.1f}), "
            f"starving the generator ({score_summary} < {f1_threshold:.2f}).{rationale_suffix}"
        )
    elif retrieval_ok and not answer_ok:
        verdict = "GENERATION_FAILURE"
        root_cause = (
            f"Generation Failure: Retriever supplied sufficient context "
            f"(Context Recall: {context_recall:.2f} >= {recall_threshold:.1f}), "
            f"but the generator failed reasoning or hallucinated ({score_summary} < {f1_threshold:.2f}).{rationale_suffix}"
        )
    else:
        verdict = "PARAMETRIC_LEAKAGE"
        root_cause = (
            f"Parametric Memory Leakage: Retriever missed gold documents "
            f"(Context Recall: {context_recall:.2f} < {recall_threshold:.1f}), "
            f"but the generator still produced the correct answer from pre-trained weights "
            f"({score_summary} >= {f1_threshold:.2f}).{rationale_suffix}"
        )

    return {
        "verdict": verdict,
        "root_cause": root_cause,
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
        "answer_f1": round(answer_f1, 4),
        "exact_match": round(exact_match, 4),
    }