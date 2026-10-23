import argparse
import pathlib
import tempfile
import unittest
from unittest import mock

from backend.benchmark_repair_effectiveness import (
    _backup_outputs,
    _request_with_retries,
    _upgrade_detail_schema,
    make_detail_row,
    summarize,
)


def response(perturbation, repair=None, correctness=0.0, faithfulness=0.0,
             f1=0.0, judge="ollama/qwen3.5:9b", generation="ollama/qwen3.5:9b",
             warning=None):
    changes = [{"method": perturbation, "status": "applied"}]
    if repair:
        changes.append({"method": repair, "status": "applied"})
    return {"variants": [{
        "methods": [perturbation], "changes": changes,
        "metrics": {
            "judge_correctness_perturbed": correctness,
            "judge_faithfulness_perturbed": faithfulness,
            "judge_model": judge, "answer_f1_perturbed": f1,
        },
        "perturbed_generation_source": generation, "warning": warning,
        "perturbed_answer": "answer", "ground_truth": "truth",
    }]}


class RepairEffectivenessBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            llm_model="ollama/qwen3.5:9b", judge_model="ollama/qwen3.5:9b",
            embedding_provider="embed", search_type="similarity", top_k=2,
            typo_rate=0.25,
        )
        self.question = {"id": "q1", "question": "Question?", "ground_truth": "truth"}

    def test_row_uses_judge_threshold_and_marks_recovery(self):
        row = make_detail_row(
            self.question, "shuffle", "fix_shuffle", 42,
            response("shuffle", correctness=0.49, faithfulness=0.6, f1=0.2),
            response("shuffle", "fix_shuffle", correctness=0.5, faithfulness=0.8, f1=0.7),
            self.args,
        )
        self.assertTrue(row["valid_for_analysis"])
        self.assertTrue(row["recovered"])
        self.assertFalse(row["degraded"])
        self.assertAlmostEqual(row["correctness_delta"], 0.01)

    def test_synthetic_judge_fallback_is_flagged_and_excluded(self):
        row = make_detail_row(
            self.question, "conflict", "contradiction_filter", 42,
            response("conflict", judge="synthetic-evaluator"),
            response("conflict", "contradiction_filter", correctness=1.0), self.args,
        )
        self.assertFalse(row["valid_for_analysis"])
        self.assertIn("synthetic judge fallback", row["exclusion_reason"])
        self.assertEqual(row["correct_off"], "")

    def test_skipped_repair_is_reported_but_not_mislabelled_as_missing(self):
        on = response("alter_facts", "fix_alter_facts", correctness=1.0)
        on["variants"][0]["changes"][-1].update({
            "status": "skipped", "summary": "No cross-checkable figures disagreed."
        })
        row = make_detail_row(
            self.question, "alter_facts", "fix_alter_facts", 42,
            response("alter_facts", correctness=1.0), on, self.args,
        )
        self.assertTrue(row["valid_for_analysis"])
        self.assertFalse(row["repair_applied"])
        self.assertEqual(row["repair_status"], "skipped")
        self.assertIn("No cross-checkable", row["repair_summary"])

    def test_missing_repair_record_is_excluded(self):
        row = make_detail_row(
            self.question, "conflict", "contradiction_filter", 42,
            response("conflict"), response("conflict"), self.args,
        )
        self.assertFalse(row["valid_for_analysis"])
        self.assertEqual(row["repair_status"], "missing")
        self.assertIn("missing repair result", row["exclusion_reason"])

    @mock.patch("backend.benchmark_repair_effectiveness.time.sleep")
    @mock.patch("backend.benchmark_repair_effectiveness.http_json")
    def test_fallback_response_is_retried(self, http_json, sleep):
        fallback = response("conflict", generation="synthetic",
                            warning="Ollama returned empty response. Used synthetic fallback.")
        genuine = response("conflict")
        http_json.side_effect = [fallback, genuine]
        result = _request_with_retries("url", {}, 10, 3, 0.25)
        self.assertIs(result, genuine)
        self.assertEqual(http_json.call_count, 2)
        sleep.assert_called_once_with(0.25)

    @mock.patch("backend.benchmark_repair_effectiveness.http_json")
    def test_exhausted_fallback_retries_return_flagged_result(self, http_json):
        fallback = response("conflict", judge="synthetic-evaluator", generation="synthetic")
        http_json.return_value = fallback
        result = _request_with_retries("url", {}, 10, 2, 0)
        self.assertIs(result, fallback)
        self.assertEqual(http_json.call_count, 2)

    @mock.patch("backend.benchmark_repair_effectiveness.http_json")
    def test_missing_variants_response_is_retried(self, http_json):
        genuine = response("conflict")
        http_json.side_effect = [{}, genuine]
        result = _request_with_retries("url", {}, 10, 2, 0)
        self.assertIs(result, genuine)

    def test_no_resume_backups_preserve_prior_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            details = pathlib.Path(directory) / "details.csv"
            summary = pathlib.Path(directory) / "summary.csv"
            details.write_text("old details", encoding="utf-8")
            summary.write_text("old summary", encoding="utf-8")
            backups = _backup_outputs((details, summary))
            self.assertFalse(details.exists())
            self.assertFalse(summary.exists())
            self.assertEqual([backup.read_text(encoding="utf-8") for _, backup in backups],
                             ["old details", "old summary"])

    def test_resume_upgrades_old_detail_schema_with_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            details = pathlib.Path(directory) / "details.csv"
            details.write_text("question_id,repair_applied\nq1,False\n", encoding="utf-8")
            rows = [{"question_id": "q1", "repair_applied": "False"}]
            backup = _upgrade_detail_schema(details, rows)
            self.assertIsNotNone(backup)
            self.assertTrue(backup.exists())
            self.assertEqual(rows[0]["repair_status"], "legacy_unknown")
            self.assertIn("repair_status", details.read_text(encoding="utf-8-sig").splitlines()[0])

    def test_summary_reports_accuracy_delta_recovered_and_degraded(self):
        base = {"perturbation": "truncate", "repair": "fix_truncate",
                "valid_for_analysis": True, "faithfulness_off": 0.5,
                "faithfulness_on": 0.6, "faithfulness_delta": 0.1,
                "f1_off": 0.4, "f1_on": 0.5, "f1_delta": 0.1,
                "correctness_off": 0.4, "correctness_on": 0.6,
                "correctness_delta": 0.2}
        rows = [dict(base, correct_off=False, correct_on=True, recovered=True, degraded=False),
                dict(base, correct_off=True, correct_on=False, recovered=False, degraded=True),
                dict(base, valid_for_analysis=False, correct_off="", correct_on="",
                     recovered=False, degraded=False)]
        result = summarize(rows)[0]
        self.assertEqual(result["rows_valid"], 2)
        self.assertEqual(result["rows_excluded"], 1)
        self.assertEqual(result["accuracy_delta"], 0.0)
        self.assertEqual(result["recovered_count"], 1)
        self.assertEqual(result["degraded_count"], 1)


if __name__ == "__main__":
    unittest.main()
