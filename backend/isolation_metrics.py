"""
Stage Isolation Metrics and Root-Cause Failure Attribution Engine.

Provides modular evaluation utilities to cleanly isolate whether a RAG failure
originates in the Retrieval Stage (omitted facts) or the Generation Stage
(hallucination or reasoning breakdown), or if the model succeeded via
Parametric Memory Leakage.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional


def normalize_text(text: Optional[str]) -> str:
    """
    Standard text normalization for lexical evaluation:
    - Lowercases text
    - Strips all punctuation
    - Normalizes whitespace
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Collapse whitespace
    return " ".join(text.split())


def calculate_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes token-level F1 score between prediction and ground truth.
    
    Tokens are lowercased and punctuation-stripped.
    Handles edge cases:
      - If both prediction and ground truth normalize to empty: returns 1.0
      - If only one is empty: returns 0.0
      - Harmonic mean of token precision and recall
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)

    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common_counts = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common_counts.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(round(f1, 4))


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Computes binary exact match (1.0 or 0.0) after text normalization.
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def calculate_context_recall(retrieved_titles: List[str], gold_titles: List[str]) -> float:
    """
    Calculates Context Recall:
        Recall = |retrieved_titles ∩ gold_titles| / |gold_titles|
    Returns 1.0 if gold_titles is empty.
    """
    if not gold_titles:
        return 1.0
    
    gold_set = set(gold_titles)
    retrieved_set = set(retrieved_titles)
    hits = len(retrieved_set & gold_set)
    recall = hits / len(gold_set)
    return float(round(recall, 4))


def calculate_context_precision(retrieved_titles: List[str], gold_titles: List[str]) -> float:
    """
    Evaluates whether the gold documents appear at the top ranks versus being buried
    under distractors using standard Mean Average Precision (MAP) at K:
        Precision = (Σ_{k=1}^K Precision@k * v_k) / |retrieved ∩ gold|
    where v_k = 1 if the k-th retrieved document is in gold_titles, else 0.
    
    Returns:
      - 1.0 if gold_titles is empty.
      - 0.0 if no gold documents are retrieved.
    """
    if not gold_titles:
        return 1.0
    if not retrieved_titles:
        return 0.0

    gold_set = set(gold_titles)
    hits = 0
    precisions_sum = 0.0

    for rank, title in enumerate(retrieved_titles, start=1):
        if title in gold_set:
            hits += 1
            precisions_sum += hits / rank

    if hits == 0:
        return 0.0

    precision = precisions_sum / hits
    return float(round(precision, 4))


def diagnose_failure(
    retrieved_titles: List[str],
    gold_titles: List[str],
    generated_answer: str,
    ground_truth: str,
    f1_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Diagnoses pipeline execution against the 4 diagnostic quadrants:
      - SUCCESS: Context Recall >= threshold AND Answer F1 >= threshold.
      - RETRIEVAL_FAILURE: Context Recall < threshold AND Answer F1 < threshold
          (Cascading Failure: Retriever starved generator of facts).
      - GENERATION_FAILURE: Context Recall >= threshold AND Answer F1 < threshold
          (Retriever provided necessary context, but LLM hallucinated or failed reasoning).
      - PARAMETRIC_LEAKAGE: Context Recall < threshold AND Answer F1 >= threshold
          (Retriever failed, but LLM answered correctly from pre-trained parametric weights).

    Threshold for Context Recall:
      - Single-hop (len(gold_titles) <= 1): requires 1.0 recall.
      - Multi-hop (len(gold_titles) > 1): requires >= 0.5 recall.
    """
    context_recall = calculate_context_recall(retrieved_titles, gold_titles)
    context_precision = calculate_context_precision(retrieved_titles, gold_titles)
    answer_f1 = calculate_token_f1(generated_answer, ground_truth)

    # Determine recall threshold based on single-hop vs multi-hop
    recall_threshold = 1.0 if len(gold_titles) <= 1 else 0.5
    recall_pass = context_recall >= recall_threshold
    f1_pass = answer_f1 >= f1_threshold

    if recall_pass and f1_pass:
        verdict = "SUCCESS"
        root_cause = (
            f"Pipeline succeeded: Retriever supplied necessary context "
            f"(Context Recall: {context_recall:.2f}, Precision: {context_precision:.2f}) "
            f"and LLM answered accurately (Token F1: {answer_f1:.2f})."
        )
    elif not recall_pass and not f1_pass:
        verdict = "RETRIEVAL_FAILURE"
        root_cause = (
            f"Cascading Retrieval Failure: Retriever failed to provide necessary gold documents "
            f"(Context Recall: {context_recall:.2f} < {recall_threshold:.1f}), "
            f"starving the generator of facts required to produce an accurate answer (Token F1: {answer_f1:.2f})."
        )
    elif recall_pass and not f1_pass:
        verdict = "GENERATION_FAILURE"
        root_cause = (
            f"Generation Failure: Retriever supplied sufficient context "
            f"(Context Recall: {context_recall:.2f} >= {recall_threshold:.1f}), "
            f"but the generator failed reasoning or hallucinated (Token F1: {answer_f1:.2f} < {f1_threshold:.2f})."
        )
    else:  # not recall_pass and f1_pass
        verdict = "PARAMETRIC_LEAKAGE"
        root_cause = (
            f"Parametric Memory Leakage: Retriever missed required evidence "
            f"(Context Recall: {context_recall:.2f} < {recall_threshold:.1f}), "
            f"yet the LLM produced a correct answer (Token F1: {answer_f1:.2f}) from internal pre-trained memory."
        )

    return {
        "context_recall": context_recall,
        "context_precision": context_precision,
        "answer_f1": answer_f1,
        "verdict": verdict,
        "root_cause": root_cause
    }
