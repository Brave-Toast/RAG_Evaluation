import copy
import unittest

from backend.perturbations import (
    METHOD_ORDER, PerturbationSettings, apply_context_methods, distractor_pool,
)


class ContextPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"title": "First", "text": "Founded in 1844 by a local publisher.", "score": 0.9},
            {"title": "Second", "text": "Started in 1989 and published every month.", "score": 0.8},
            {"title": "Third", "text": "This passage has no numbers at all.", "score": 0.7},
        ]

    def apply(self, methods, **kwargs):
        return apply_context_methods(self.docs, methods, PerturbationSettings(), 42, **kwargs)

    def test_removal_uses_annotations_and_reports_unverified_fallback(self):
        docs, trace = self.apply(["remove_evidence"], supporting_titles=["Second"])
        self.assertEqual([d["title"] for d in docs], ["First", "Third"])
        self.assertIn("annotated", trace[0]["summary"])
        docs, trace = self.apply(["remove_evidence"])
        self.assertEqual([d["title"] for d in docs], ["Second", "Third"])
        self.assertIn("support annotations unavailable", trace[0]["summary"])
        docs, trace = self.apply(["remove_evidence"], supporting_titles=["Not retrieved"])
        self.assertEqual(docs, self.docs)
        self.assertEqual(trace[0]["status"], "skipped")

    def test_numeric_edits_preserve_inputs_and_log_actual_replacements(self):
        original = copy.deepcopy(self.docs)
        docs, trace = self.apply(["alter_facts"])
        edit = trace[0]["details"][0]
        self.assertNotEqual(edit["original"], edit["replacement"])
        self.assertIn(edit["replacement"], docs[edit["index"]]["text"])
        self.assertIsNone(docs[edit["index"]]["score"])
        self.assertEqual(self.docs, original)
        self.assertEqual(self.apply(["alter_facts"]), (docs, trace))

    def test_fact_count_and_missing_tokens(self):
        docs, trace = apply_context_methods(self.docs, ["alter_facts"], PerturbationSettings(fact_count=20), 4)
        self.assertEqual(len(trace[0]["details"]), 2)
        _, trace = apply_context_methods([self.docs[2]], ["alter_facts"], PerturbationSettings(), 4)
        self.assertEqual(trace[0]["status"], "skipped")

    def test_truncation_keeps_word_boundary_and_reports_full_retention_noop(self):
        docs, trace = self.apply(["truncate"])
        for before, after in zip(self.docs, docs):
            self.assertTrue(before["text"].startswith(after["text"]))
            self.assertLessEqual(len(after["text"]), len(before["text"]) // 2)
            if after["text"]:
                self.assertTrue(before["text"][len(after["text"])].isspace())
        docs, trace = apply_context_methods(self.docs, ["truncate"], PerturbationSettings(retain_percent=100), 4)
        self.assertEqual(docs, self.docs)
        self.assertEqual(trace[0]["status"], "skipped")

    def test_shuffle_changes_order_only_and_handles_singletons(self):
        docs, trace = self.apply(["shuffle"])
        self.assertNotEqual(docs, self.docs)
        self.assertCountEqual(docs, self.docs)
        self.assertEqual(self.apply(["shuffle"]), (docs, trace))
        _, trace = apply_context_methods(self.docs[:1], ["shuffle"], PerturbationSettings(), 0)
        self.assertEqual(trace[0]["status"], "skipped")

    def test_distractors_exclude_other_question_duplicates_and_original_candidates(self):
        dataset = [
            {"id": "current", "contexts": [{"title": "Same question", "text": "Exclude"}]},
            {"id": "other", "contexts": self.docs + [{"title": "New", "text": "Different subject"}]},
            {"id": "duplicate", "contexts": [{"title": "NEW", "text": "Another text"},
                                                {"title": "Different title", "text": "different subject"}]},
        ]
        pool = distractor_pool(dataset, "current", self.docs)
        self.assertEqual(len(pool), 1)
        docs, trace = self.apply(["irrelevant"], distractors=pool)
        self.assertEqual(docs[:-1], self.docs)
        self.assertEqual(trace[0]["details"][0]["source_question_id"], "other")

    def test_composition_order_and_no_mutation_of_nested_inputs(self):
        original = copy.deepcopy(self.docs)
        pool = [{"title": "Distractor", "text": "A long untouched passage from 2020.", "source_question_id": "other"}]
        docs, trace = self.apply(list(reversed(METHOD_ORDER)), distractors=pool)
        self.assertEqual([c["method"] for c in trace], ["remove_evidence", "alter_facts", "truncate", "irrelevant", "shuffle"])
        self.assertEqual(next(d["text"] for d in docs if d["title"] == "Distractor"), pool[0]["text"])
        self.assertNotIn("First", [d["title"] for d in docs])
        self.assertEqual(self.docs, original)

    def test_empty_context_and_removing_every_passage(self):
        docs, trace = apply_context_methods([], METHOD_ORDER, PerturbationSettings(), 0)
        self.assertEqual(docs, [])
        self.assertTrue(all(change["status"] == "skipped" for change in trace))
        docs, _ = apply_context_methods(self.docs, ["remove_evidence"], PerturbationSettings(remove_count=5), 0)
        self.assertEqual(docs, [])


if __name__ == "__main__":
    unittest.main()
