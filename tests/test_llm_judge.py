"""Unit and API integration tests for the LLM-as-a-Judge evaluation stage."""

import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from backend.app import app
from backend.isolation_metrics import diagnose_failure
from backend.llm_judge import (
    _extract_json_object,
    evaluate_synthetic_judge,
    register_llm_judge,
)

register_llm_judge(app)


class LLMJudgeUnitTests(unittest.TestCase):
    def test_verbose_sentence_resolves_to_success_with_judge(self):
        contexts = [
            {
                "title": "The Oberoi Group",
                "text": "The Oberoi Group is a hotel company with its head office in Delhi.",
            }
        ]
        ans = "The Oberoi Group has its head office in Delhi."
        gt = "Delhi"

        # Without judge: lexical F1 is 0.20 -> GENERATION_FAILURE
        lexical_diag = diagnose_failure(["The Oberoi Group"], ["The Oberoi Group"], ans, gt)
        self.assertEqual(lexical_diag["verdict"], "GENERATION_FAILURE")

        # With judge: semantic containment recognizes "Delhi" -> SUCCESS
        judge_res = evaluate_synthetic_judge(
            "The Oberoi family is part of a hotel company that has a head office in what city?",
            contexts,
            ans,
            gt,
        )
        self.assertGreaterEqual(judge_res["correctness"], 0.9)
        self.assertGreaterEqual(judge_res["faithfulness"], 0.5)

        judge_diag = diagnose_failure(
            ["The Oberoi Group"], ["The Oberoi Group"], ans, gt, judge_result=judge_res
        )
        self.assertEqual(judge_diag["verdict"], "SUCCESS")
        self.assertIn("Judge Correctness", judge_diag["root_cause"])

    def test_extracts_json_from_thinking_model_output(self):
        raw = (
            "<think>\nComparing answer with ground truth...\n</think>\n"
            '{"correctness": 1.0, "faithfulness": 0.95, "context_relevance": 1.0, '
            '"verdict": "PASS", "rationale": "Exact semantic match grounded in context."}'
        )
        parsed = _extract_json_object(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["verdict"], "PASS")
        self.assertEqual(parsed["correctness"], 1.0)


class LLMJudgeApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.payload = {
            "question_id": "5a7a06935542990198eaf050",
            "baseline_query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "ground_truth": "Arthur's Magazine",
            "gold_titles": ["Arthur's Magazine", "First for Women"],
            "contexts": [
                {
                    "title": "Arthur's Magazine",
                    "text": "Arthur's Magazine (1844-1846) was an American literary periodical.",
                },
                {
                    "title": "First for Women",
                    "text": "First for Women is a woman's magazine published in 1989.",
                },
            ],
            "perturbation_type": "spelling",
            "typo_rate": 0.0,
            "top_k": 2,
            "llm_model": "synthetic-evaluator",
            "use_llm_judge": True,
            "judge_model": "synthetic-evaluator",
        }

    def test_judges_list_endpoint(self):
        res = self.client.get("/api/judges")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("default", data)
        self.assertGreaterEqual(len(data["judges"]), 3)

    def test_api_populates_judge_metrics_when_enabled(self):
        res = self.client.post("/api/run-evaluation", json=self.payload)
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        self.assertTrue(metrics["use_llm_judge"])
        self.assertEqual(metrics["judge_model"], "synthetic-evaluator")
        self.assertGreaterEqual(metrics["judge_correctness_baseline"], 0.9)
        self.assertGreaterEqual(metrics["judge_faithfulness_baseline"], 0.5)
        self.assertEqual(metrics["judge_verdict_perturbed"], "PASS")
        self.assertEqual(metrics["failure_attribution"], "SUCCESS")

    def test_api_rejects_unknown_judge_model(self):
        res = self.client.post(
            "/api/run-evaluation",
            json={**self.payload, "judge_model": "invalid/model"},
        )
        self.assertEqual(res.status_code, 422)

    def test_api_falls_back_gracefully_when_ollama_judge_offline(self):
        with patch(
            "backend.llm_judge.httpx.AsyncClient.post",
            new=AsyncMock(side_effect=RuntimeError("connection refused")),
        ):
            res = self.client.post(
                "/api/run-evaluation",
                json={**self.payload, "judge_model": "ollama/qwen3.5:9b"},
            )
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        self.assertEqual(metrics["judge_model"], "synthetic-evaluator")
        self.assertIsNotNone(metrics["judge_rationale_perturbed"])


if __name__ == "__main__":
    unittest.main()