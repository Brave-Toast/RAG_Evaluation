import unittest

from backend.benchmark_fix_spelling_answers import _score, summarize


class FixSpellingAnswerBenchmarkTests(unittest.TestCase):
    def test_score_reuses_project_normalized_f1_and_exact_match(self):
        score = _score("Arthur's Magazine.", "arthurs magazine")
        self.assertEqual(score["token_f1"], 1.0)
        self.assertTrue(score["exact_match"])
        self.assertTrue(score["correct"])

    def test_summary_counts_recoveries_and_regressions(self):
        def condition(correct):
            return {"correct": correct}
        rows = [
            {"clean": condition(True), "broken": condition(False), "fixed": condition(True)},
            {"clean": condition(True), "broken": condition(True), "fixed": condition(False)},
            {"clean": condition(True), "broken": condition(False), "fixed": condition(False)},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["broken_incorrect_recovered_by_fixed"], 1)
        self.assertEqual(summary["broken_correct_worsened_by_fixed"], 1)
        self.assertEqual(summary["overall_recovery_percentage"], 50.0)


if __name__ == "__main__":
    unittest.main()
