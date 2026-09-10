"""
End-to-End Scenario Verification for the 4 Attribution Quadrants.
Tests live execution through the FastAPI /api/run-evaluation pipeline.
"""

import unittest
from backend.app import app
from fastapi.testclient import TestClient


class AttributionScenariosVerification(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.base_contexts = [
            {
                "title": "Arthur's Magazine",
                "text": "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century."
            },
            {
                "title": "First for Women",
                "text": "First for Women is a woman's magazine published in 1989."
            },
            {
                "title": "Radio City (Indian radio station)",
                "text": "Radio City is India's first private FM radio station and was started on 3 July 2001."
            }
        ]

    def test_scenario_1_clean_unperturbed_success(self):
        """Unperturbed query retrieves gold titles and answers correctly -> SUCCESS."""
        payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "gold_titles": ["Arthur's Magazine", "First for Women"],
            "contexts": self.base_contexts,
            "perturbation_type": "spelling",
            "typo_rate": 0.0,
            "top_k": 2,
            "llm_model": "synthetic-evaluator"
        }
        res = self.client.post("/api/run-evaluation", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        self.assertGreaterEqual(metrics["context_recall_baseline"], 0.5)
        self.assertGreaterEqual(metrics["answer_f1_baseline"], 0.5)
        self.assertEqual(metrics["failure_attribution"], "SUCCESS")
        self.assertIn("Pipeline succeeded", metrics["failure_root_cause"])

    def test_scenario_2_generation_failure_under_conflict(self):
        """Gold documents retrieved, but LLM fails answer -> GENERATION_FAILURE."""
        payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "gold_titles": ["Arthur's Magazine", "First for Women"],
            "contexts": self.base_contexts,
            "perturbation_type": "conflict",
            "top_k": 3,
            "llm_model": "synthetic-evaluator"
        }
        res = self.client.post("/api/run-evaluation", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        # In conflict mode, synthetic evaluator outputs 'Conflicting sources refute...'
        self.assertGreaterEqual(metrics["context_recall_perturbed"], 0.5)
        self.assertLess(metrics["answer_f1_perturbed"], 0.5)
        self.assertEqual(metrics["failure_attribution"], "GENERATION_FAILURE")
        self.assertIn("Generation Failure", metrics["failure_root_cause"])

    def test_scenario_3_retrieval_failure_when_gold_dropped(self):
        """Evidence removed, retriever starves generator -> RETRIEVAL_FAILURE."""
        payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "gold_titles": ["Arthur's Magazine", "First for Women"],
            "contexts": [
                {"title": "Radio City (Indian radio station)", "text": "Radio station founded in India in 2001."},
                {"title": "History of Albanian football", "text": "Albanian football existed before the federation."}
            ],
            # Pass non-gold contexts with synthetic evaluator returning fallback non-matching text
            "custom_query": "Which magazine was started first?",
            "perturbation_type": "conflict",
            "top_k": 2,
            "llm_model": "synthetic-evaluator"
        }
        res = self.client.post("/api/run-evaluation", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        self.assertLess(metrics["context_recall_perturbed"], 0.5)
        self.assertLess(metrics["answer_f1_perturbed"], 0.5)
        self.assertEqual(metrics["failure_attribution"], "RETRIEVAL_FAILURE")
        self.assertIn("Cascading Retrieval Failure", metrics["failure_root_cause"])

    def test_scenario_4_parametric_leakage(self):
        """Retriever misses gold documents, but generator answers correctly -> PARAMETRIC_LEAKAGE."""
        payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "gold_titles": ["Arthur's Magazine", "First for Women"],
            # Distractor contexts only -> recall is 0.0
            "contexts": [
                {"title": "Radio City", "text": "Indian radio station."},
                {"title": "Albanian football", "text": "Football in Albania."}
            ],
            "perturbation_type": "spelling",
            "typo_rate": 0.0,
            "top_k": 2,
            "llm_model": "synthetic-evaluator"
        }
        # Synthetic evaluator unperturbed outputs ground truth ('Arthur's Magazine')
        res = self.client.post("/api/run-evaluation", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        self.assertLess(metrics["context_recall_perturbed"], 0.5)
        self.assertGreaterEqual(metrics["answer_f1_perturbed"], 0.5)
        self.assertEqual(metrics["failure_attribution"], "PARAMETRIC_LEAKAGE")
        self.assertIn("Parametric Memory Leakage", metrics["failure_root_cause"])


if __name__ == "__main__":
    unittest.main()
