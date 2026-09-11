"""
Unit and Integration Tests for Stage Isolation Metrics and Root-Cause Failure Attribution.
"""

import copy
import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

from backend.isolation_metrics import (
    normalize_text,
    calculate_token_f1,
    calculate_exact_match,
    calculate_context_recall,
    calculate_context_precision,
    diagnose_failure
)
from backend.app import app, STAGE_LOG_PATH
from fastapi.testclient import TestClient


class IsolationMetricsUnitTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Arthur's Magazine!  "), "arthurs magazine")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("UPPERCASE, with-hyphens."), "uppercase withhyphens")

    def test_calculate_token_f1(self):
        # Empty string handling
        self.assertEqual(calculate_token_f1("", ""), 1.0)
        self.assertEqual(calculate_token_f1("hello", ""), 0.0)
        self.assertEqual(calculate_token_f1("", "world"), 0.0)
        self.assertEqual(calculate_token_f1(None, None), 1.0)

        # Exact and punctuation-insensitive matches
        self.assertEqual(calculate_token_f1("Arthur's Magazine", "Arthur's Magazine"), 1.0)
        self.assertEqual(calculate_token_f1("Arthur's Magazine", "Arthurs Magazine"), 1.0)
        self.assertEqual(calculate_token_f1("THE OBEROI GROUP", "the oberoi group"), 1.0)

        # Partial token matches
        # Prediction: "first for women" (3 tokens) | GT: "first" (1 token)
        # Precision = 1/3, Recall = 1/1, F1 = 2 * (1/3 * 1) / (1/3 + 1) = 2/3 / (4/3) = 0.5
        self.assertAlmostEqual(calculate_token_f1("first for women", "first"), 0.5, places=3)

        # Completely disjoint
        self.assertEqual(calculate_token_f1("completely different answer", "Arthur's Magazine"), 0.0)

    def test_calculate_exact_match(self):
        self.assertEqual(calculate_exact_match("Arthur's Magazine", "arthurs magazine"), 1.0)
        self.assertEqual(calculate_exact_match("Arthur's Magazine", "Arthur"), 0.0)
        self.assertEqual(calculate_exact_match("", ""), 1.0)

    def test_calculate_context_recall(self):
        # Empty gold titles edge case: returns 1.0
        self.assertEqual(calculate_context_recall([], []), 1.0)
        self.assertEqual(calculate_context_recall(["Doc 1"], []), 1.0)

        # Multi-hop retrieval scenarios
        gold = ["Arthur's Magazine", "First for Women"]
        self.assertEqual(calculate_context_recall(["Arthur's Magazine", "First for Women"], gold), 1.0)
        self.assertEqual(calculate_context_recall(["Arthur's Magazine", "Distractor"], gold), 0.5)
        self.assertEqual(calculate_context_recall(["Distractor 1", "Distractor 2"], gold), 0.0)

        # Single-hop retrieval
        single_gold = ["Hoover Dam"]
        self.assertEqual(calculate_context_recall(["Hoover Dam", "Other"], single_gold), 1.0)
        self.assertEqual(calculate_context_recall(["Other"], single_gold), 0.0)

    def test_calculate_context_precision(self):
        # Empty gold titles: returns 1.0
        self.assertEqual(calculate_context_precision([], []), 1.0)
        self.assertEqual(calculate_context_precision(["Doc 1"], []), 1.0)

        # Zero gold documents retrieved: returns 0.0
        self.assertEqual(calculate_context_precision(["Distractor 1", "Distractor 2"], ["Gold 1"]), 0.0)

        # Gold document at rank 1: Precision@1 = 1/1 = 1.0
        self.assertEqual(calculate_context_precision(["Gold 1", "Distractor"], ["Gold 1"]), 1.0)

        # Gold document buried at rank 2: Precision@2 = 1/2 = 0.5
        self.assertEqual(calculate_context_precision(["Distractor", "Gold 1"], ["Gold 1"]), 0.5)

        # Both gold documents at ranks 1 and 2: Precision@1 = 1.0, Precision@2 = 1.0 -> Avg = 1.0
        self.assertEqual(calculate_context_precision(["Gold 1", "Gold 2"], ["Gold 1", "Gold 2"]), 1.0)

        # Gold documents at ranks 2 and 3: Precision@2 = 1/2 = 0.5, Precision@3 = 2/3 = 0.6667 -> Avg = 0.5833
        self.assertAlmostEqual(
            calculate_context_precision(["Distractor", "Gold 1", "Gold 2"], ["Gold 1", "Gold 2"]),
            (0.5 + 2/3) / 2,
            places=3
        )

    def test_diagnose_failure_quadrants(self):
        gold = ["Arthur's Magazine", "First for Women"]
        gt = "Arthur's Magazine"

        # 1. SUCCESS: Recall >= 0.5 and F1 >= 0.5
        res_success = diagnose_failure(
            retrieved_titles=["Arthur's Magazine", "First for Women"],
            gold_titles=gold,
            generated_answer="Arthur's Magazine was started first.",
            ground_truth=gt,
            f1_threshold=0.5
        )
        self.assertEqual(res_success["verdict"], "SUCCESS")
        self.assertIn("Pipeline succeeded", res_success["root_cause"])
        self.assertGreaterEqual(res_success["context_recall"], 0.5)
        self.assertGreaterEqual(res_success["answer_f1"], 0.5)

        # 2. RETRIEVAL_FAILURE: Recall < 0.5 and F1 < 0.5 (Cascading failure)
        res_ret_fail = diagnose_failure(
            retrieved_titles=["Radio City", "Echosmith"],
            gold_titles=gold,
            generated_answer="Echosmith started first.",
            ground_truth=gt,
            f1_threshold=0.5
        )
        self.assertEqual(res_ret_fail["verdict"], "RETRIEVAL_FAILURE")
        self.assertIn("Cascading Retrieval Failure", res_ret_fail["root_cause"])
        self.assertLess(res_ret_fail["context_recall"], 0.5)
        self.assertLess(res_ret_fail["answer_f1"], 0.5)

        # 3. GENERATION_FAILURE: Recall >= 0.5 and F1 < 0.5 (LLM hallucination / breakdown)
        res_gen_fail = diagnose_failure(
            retrieved_titles=["Arthur's Magazine", "First for Women"],
            gold_titles=gold,
            generated_answer="Conflicting sources refute that Arthur's Magazine was started first.",
            ground_truth=gt,
            f1_threshold=0.5
        )
        self.assertEqual(res_gen_fail["verdict"], "GENERATION_FAILURE")
        self.assertIn("Generation Failure", res_gen_fail["root_cause"])
        self.assertGreaterEqual(res_gen_fail["context_recall"], 0.5)
        self.assertLess(res_gen_fail["answer_f1"], 0.5)

        # 4. PARAMETRIC_LEAKAGE: Recall < 0.5 and F1 >= 0.5 (Model answered from weights)
        res_leak = diagnose_failure(
            retrieved_titles=["Radio City", "Echosmith"],
            gold_titles=gold,
            generated_answer="Arthur's Magazine",
            ground_truth=gt,
            f1_threshold=0.5
        )
        self.assertEqual(res_leak["verdict"], "PARAMETRIC_LEAKAGE")
        self.assertIn("Parametric Memory Leakage", res_leak["root_cause"])
        self.assertLess(res_leak["context_recall"], 0.5)
        self.assertGreaterEqual(res_leak["answer_f1"], 0.5)

    def test_single_hop_threshold_behavior(self):
        single_gold = ["Hoover Dam"]
        gt = "Hydroelectric power"

        # Single-hop requires Recall == 1.0 (since <= 1 gold title)
        # If retriever retrieved nothing relevant and answer is wrong, recall = 0.0 -> RETRIEVAL_FAILURE
        res = diagnose_failure(
            retrieved_titles=["Clean Energy Overview"],
            gold_titles=single_gold,
            generated_answer="Nuclear energy",
            ground_truth=gt
        )
        self.assertEqual(res["verdict"], "RETRIEVAL_FAILURE")

        # When single gold is retrieved -> SUCCESS
        res_ok = diagnose_failure(
            retrieved_titles=["Hoover Dam"],
            gold_titles=single_gold,
            generated_answer="Hydroelectric power",
            ground_truth=gt
        )
        self.assertEqual(res_ok["verdict"], "SUCCESS")


class PipelineIsolationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = pathlib.Path(self.temp_dir.name) / "test_stage_run_logs.jsonl"
        self.client = TestClient(app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_api_run_evaluation_populates_isolation_metrics(self):
        payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "gold_titles": ["Arthur's Magazine", "First for Women"],
            "contexts": [
                {"title": "Arthur's Magazine", "text": "Arthur's Magazine (1844-1846) was published in Philadelphia."},
                {"title": "First for Women", "text": "First for Women was started in 1989."},
                {"title": "Distractor", "text": "Unrelated historical notes."}
            ],
            "perturbation_type": "spelling",
            "typo_rate": 0.0,
            "top_k": 2,
            "llm_model": "synthetic-evaluator"
        }

        with patch("backend.app.STAGE_LOG_PATH", self.log_file):
            response = self.client.post("/api/run-evaluation", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()

            # Verify metrics contain all stage isolation fields
            metrics = data["metrics"]
            self.assertIn("context_recall_baseline", metrics)
            self.assertIn("context_recall_perturbed", metrics)
            self.assertIn("answer_f1_baseline", metrics)
            self.assertIn("answer_f1_perturbed", metrics)
            self.assertIn("failure_attribution", metrics)
            self.assertIn("failure_root_cause", metrics)

            # Synthetic baseline should match ground truth -> SUCCESS
            self.assertEqual(metrics["failure_attribution"], "SUCCESS")
            self.assertGreaterEqual(metrics["context_recall_baseline"], 0.5)
            self.assertEqual(metrics["answer_f1_baseline"], 1.0)

            # Verify stage_run_logs.jsonl contains logged fields
            self.assertTrue(self.log_file.exists())
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertGreater(len(lines), 0)
            log_record = lines[0]
            self.assertIn("gold_titles", log_record)
            self.assertEqual(log_record["gold_titles"], ["Arthur's Magazine", "First for Women"])
            self.assertIn("context_recall", log_record)
            self.assertIn("failure_attribution", log_record)
            self.assertEqual(log_record["failure_attribution"], "SUCCESS")

    def test_api_auto_resolves_gold_titles_from_dataset(self):
        # Request omitting gold_titles: backend should resolve via question_id from hotpotqa_dataset.json
        payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "contexts": [
                {"title": "Arthur's Magazine", "text": "Arthur's Magazine (1844-1846) was published in Philadelphia."},
                {"title": "First for Women", "text": "First for Women was started in 1989."}
            ],
            "perturbation_type": "spelling",
            "typo_rate": 0.0,
            "top_k": 2,
            "llm_model": "synthetic-evaluator"
        }

        with patch("backend.app.STAGE_LOG_PATH", self.log_file):
            response = self.client.post("/api/run-evaluation", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            # Verified that context_recall is computed against the resolved gold titles
            self.assertEqual(data["metrics"]["context_recall_baseline"], 1.0)
            self.assertEqual(data["metrics"]["failure_attribution"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
