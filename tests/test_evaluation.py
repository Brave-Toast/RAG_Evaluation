"""Offline orchestration/API tests: no downloaded models or running Ollama needed."""
import copy
import importlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from fastapi.testclient import TestClient

# Isolate heavyweight startup imports without changing production model loading.
with patch.dict(sys.modules, {
    "sentence_transformers": types.SimpleNamespace(SentenceTransformer=Mock(return_value=None)),
    "langchain_chroma": types.SimpleNamespace(Chroma=Mock()),
    "langchain_core.documents": types.SimpleNamespace(Document=Mock()),
}):
    backend = importlib.import_module("backend.app")

from backend.perturbations import METHOD_ORDER, PRE_RETRIEVAL


class EnsembleRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            backend.ContextDoc(title="First", text="first text"),
            backend.ContextDoc(title="Second", text="second text"),
        ]

    def test_ensemble_fuses_component_scores_and_bypasses_chroma(self):
        vectors = {
            ("query", "sentence-transformers/all-mpnet-base-v2"): np.array([1.0, 0.0]),
            ("First first text", "sentence-transformers/all-mpnet-base-v2"): np.array([1.0, 0.0]),
            ("Second second text", "sentence-transformers/all-mpnet-base-v2"): np.array([0.0, 1.0]),
            ("query", "ollama/nomic-embed-text"): np.array([1.0, 0.0, 0.0]),
            ("First first text", "ollama/nomic-embed-text"): np.array([0.0, 1.0, 0.0]),
            ("Second second text", "ollama/nomic-embed-text"): np.array([1.0, 0.0, 0.0]),
        }
        with patch.object(backend, "compute_dense_embedding", side_effect=lambda text, provider: vectors[(text, provider)]), \
             patch.object(backend.Chroma, "from_documents") as chroma_from_documents:
            result = backend.perform_vector_retrieval(
                self.docs, "query", "similarity", 2, backend.ENSEMBLE_EMBEDDING_PROVIDER
            )

        chroma_from_documents.assert_not_called()
        self.assertEqual([doc.title for doc in result], ["First", "Second"])
        self.assertEqual([doc.score for doc in result], [0.5, 0.5])

    def test_ensemble_cosine_similarity_averages_model_scores(self):
        with patch.object(
            backend,
            "compute_dense_cosine_similarity",
            wraps=backend.compute_dense_cosine_similarity,
        ) as cosine, patch.object(
            backend,
            "compute_dense_embedding",
            side_effect=[np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                         np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        ):
            score = cosine("one", "two", backend.ENSEMBLE_EMBEDDING_PROVIDER)

        self.assertEqual(score, 0.5)
        self.assertEqual(
            [call.args[2] for call in cosine.call_args_list[1:]],
            list(backend.ENSEMBLE_COMPONENT_PROVIDERS),
        )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = pathlib.Path(self.temp.name) / "stages.jsonl"
        self.payload = {
            "question_id": "one", "baseline_query": "Which magazine started first?",
            "ground_truth": "First", "llm_model": "synthetic-evaluator", "seed": 42,
            "contexts": [{"title": "First", "text": "Started in 1844 and published monthly."},
                         {"title": "Second", "text": "Started in 1989 with many readers."}],
        }
        self.dataset = [{"id": "two", "contexts": [{"title": "Elsewhere", "text": "A completely different subject."}]}]
        self.retriever = Mock(side_effect=lambda docs, query, search, k, provider: copy.deepcopy(docs[:k]))
        # Exercise metric formulas using a cheap deterministic test encoder.
        def embed(text, provider):
            vector = np.ones(16, dtype=np.float32)
            for char in text:
                vector[ord(char) % 16] += 1
            return vector
        for patcher in [patch.object(backend, "STAGE_LOG_PATH", self.log),
                        patch.object(backend, "perform_vector_retrieval", self.retriever),
                        patch.object(backend, "compute_dense_embedding", side_effect=embed),
                        patch.object(backend, "load_hotpotqa_dataset", return_value=self.dataset)]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(backend.app)

    def run_request(self, **updates):
        response = self.client.post("/api/run-evaluation", json={**self.payload, **updates})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_legacy_request_keeps_original_response_fields_and_single_variant(self):
        for method in ["spelling", "rephrase", "conflict"]:
            result = self.run_request(perturbation_type=method)
            for key in ["baseline_query", "perturbed_query", "baseline_contexts", "perturbed_contexts",
                        "baseline_answer", "perturbed_answer", "ground_truth", "metrics", "warning"]:
                self.assertIn(key, result)
            self.assertEqual(len(result["variants"]), 1)
            self.assertEqual(result["methods"], [method])
            self.assertEqual(result["baseline_generation_source"], "synthetic")
        self.assertIn("Conflicting sources", result["perturbed_answer"])

    def test_separate_reuses_baseline_and_variants_do_not_leak(self):
        original = copy.deepcopy(self.payload)
        with patch.object(backend, "generate_llm_answer", wraps=backend.generate_llm_answer) as generate:
            result = self.run_request(perturbation_types=list(METHOD_ORDER), mode="separate")
        self.assertEqual(generate.call_count, 1 + len(METHOD_ORDER))
        self.assertEqual(sum(not call.kwargs["is_perturbed"] for call in generate.call_args_list), 1)
        self.assertEqual(self.retriever.call_count, 1 + len(PRE_RETRIEVAL))  # baseline + each pre-retrieval variant
        self.assertEqual(len(result["variants"]), len(METHOD_ORDER))
        variants = {v["methods"][0]: v for v in result["variants"]}
        baseline = result["baseline_contexts"]
        for method in ["remove_evidence", "alter_facts", "truncate", "irrelevant", "shuffle"]:
            self.assertEqual(variants[method]["perturbed_retrieved_contexts"], baseline)
            self.assertEqual(variants[method]["metrics"]["retrieval_rank_shift_jaccard"], 1)
        self.assertEqual(variants["shuffle"]["perturbed_contexts"], list(reversed(baseline)))
        self.assertEqual(variants["irrelevant"]["perturbed_contexts"][:2], baseline)
        self.assertEqual(self.payload, original)
        logs = [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len({r["evaluation_id"] for r in logs}), 1)
        self.assertEqual(len({r["variant_id"] for r in logs}), len(METHOD_ORDER))
        self.assertIn("context_stage", logs[0])
        self.assertNotIn("gemini_api_key", logs[0])

    def test_combined_order_seed_repeatability_and_original_baseline(self):
        result = self.run_request(perturbation_types=list(reversed(METHOD_ORDER)))
        again = self.run_request(perturbation_types=list(METHOD_ORDER))
        self.assertEqual(result["application_order"], list(METHOD_ORDER))
        self.assertEqual([c["method"] for c in result["changes"]], list(METHOD_ORDER))
        self.assertEqual(result["changes"], again["changes"])
        self.assertEqual(result["perturbed_contexts"], again["perturbed_contexts"])
        self.assertEqual(result["baseline_query"], self.payload["baseline_query"])
        self.assertEqual(len(result["variants"]), 1)

    def test_zero_typo_rate_and_empty_context(self):
        result = self.run_request(typo_rate=0)
        self.assertEqual(result["baseline_query"], result["perturbed_query"])
        self.assertEqual(result["changes"][0]["status"], "skipped")

    def test_fix_spelling_runs_before_variant_retrieval(self):
        real_apply = backend.apply_repair_methods

        def repair(query, contexts, methods, config=None):
            if methods == ["fix_spelling"]:
                return "fixed query", contexts, [{
                    "method": "fix_spelling", "stage": "pre_retrieval_repair",
                    "status": "applied", "summary": "fixed", "details": [],
                }]
            return real_apply(query, contexts, methods, config)

        with patch.object(backend, "apply_repair_methods", side_effect=repair):
            result = self.run_request(
                perturbation_type="spelling", repair_methods=["fix_spelling"]
            )

        self.assertEqual(self.retriever.call_args_list[1].args[1], "fixed query")
        self.assertEqual(result["perturbed_query"], "fixed query")
        self.assertEqual(result["changes"][1]["stage"], "pre_retrieval_repair")
        result = self.run_request(contexts=[], perturbation_types=["remove_evidence", "alter_facts", "shuffle", "truncate"])
        self.assertEqual(result["perturbed_contexts"], [])
        self.assertTrue(all(c["status"] == "skipped" for c in result["changes"]))

    def test_annotation_request_overrides_rank_fallback(self):
        result = self.run_request(perturbation_types=["remove_evidence"], supporting_titles=["Second"])
        self.assertEqual([d["title"] for d in result["perturbed_contexts"]], ["First"])

    def test_validation_rejects_invalid_method_mode_and_settings(self):
        cases = [{"perturbation_types": []}, {"perturbation_types": ["unknown"]}, {"mode": "other"},
                 {"settings": {"retain_percent": 0}}, {"settings": {"remove_count": -1}},
                 {"settings": {"fact_count": 1.5}}, {"settings": {"unknown": 2}},
                 {"settings": {"emoji_style": "unknown"}}, {"settings": {"emoji_position": "inside_word"}},
                 {"settings": {"emoji_count": 0}}, {"settings": {"emoji_count": 21}},
                 {"settings": {"emoji_count": 1.5}},
                 {"seed": -1}, {"typo_rate": 2}, {"top_k": 0}]
        for update in cases:
            with self.subTest(update=update):
                response = self.client.post("/api/run-evaluation", json={**self.payload, **update})
                self.assertEqual(response.status_code, 422)
        self.retriever.assert_not_called()

    def test_stream_has_progress_shared_baseline_variants_and_complete(self):
        response = self.client.post("/api/run-evaluation/stream", json={**self.payload,
                                   "perturbation_types": ["truncate", "shuffle"], "mode": "separate"})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e["event"] for e in events],
                         ["progress", "baseline", "progress", "variant", "progress", "variant", "complete"])
        self.assertEqual(events[4]["message"], "Running variant 2 of 2")
        self.assertEqual(len(events[-1]["result"]["variants"]), 2)

    def test_emoji_encoded_query_reaches_retrieval_generation_and_log(self):
        with patch.object(backend, "generate_llm_answer", wraps=backend.generate_llm_answer) as generate:
            result = self.run_request(baseline_query='before 10/09/2026', perturbation_type='emoji',
                                      settings={"emoji_style": "substitute", "emoji_count": 2})
        self.assertEqual(result['methods'], ['emoji'])
        self.assertEqual(result['perturbed_query'], 'b4️⃣ 1️⃣0️⃣/0️⃣9️⃣/2️⃣0️⃣2️⃣6️⃣')
        self.assertEqual(self.retriever.call_args_list[-1].args[1], result['perturbed_query'])
        self.assertEqual(generate.call_args_list[-1].kwargs['query'], result['perturbed_query'])
        self.assertEqual(result['baseline_query'], 'before 10/09/2026')
        logged = json.loads(self.log.read_text(encoding='utf-8'))
        self.assertEqual(logged['query_stage']['perturbed_query'], result['perturbed_query'])
        self.assertEqual(logged['settings']['emoji_style'], 'substitute')
        self.assertEqual(logged['changes'][0]['details'], result['changes'][0]['details'])

    def test_emoji_combination_occurs_after_typos_and_before_conflict(self):
        result = self.run_request(perturbation_types=['conflict', 'emoji', 'spelling'],
                                  typo_rate=1, settings={"emoji_style": "substitute", "emoji_count": 20})
        self.assertEqual(result['application_order'], ['spelling', 'emoji', 'conflict'])
        spelling, emoji, conflict = result['changes']
        self.assertEqual(emoji['input_query'], spelling['details'][0]['replacement'])
        self.assertEqual(result['perturbed_query'], emoji['output_query'])
        self.assertIn(emoji['output_query'], conflict['details'][0]['text'])

    def test_stream_failure_is_explicit_and_has_no_false_completion(self):
        self.retriever.side_effect = RuntimeError("test failure")
        response = self.client.post("/api/run-evaluation/stream", json=self.payload)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["event"], "error")
        self.assertNotIn("complete", [e["event"] for e in events])

    def test_generation_fallback_is_labelled_and_actual_output_is_identified(self):
        with patch.object(backend.httpx.AsyncClient, "post", new=AsyncMock(side_effect=backend.httpx.ConnectError("offline"))):
            result = self.run_request(llm_model="ollama/test")
        self.assertEqual(result["perturbed_generation_source"], "synthetic")
        response = Mock(status_code=200)
        response.json.return_value = {"response": "An actual model response"}
        with patch.object(backend, "ensure_ollama_model_pulled", return_value=True), \
             patch.object(backend.httpx.AsyncClient, "post", new=AsyncMock(return_value=response)):
            result = self.run_request(llm_model="ollama/test")
        self.assertEqual(result["perturbed_generation_source"], "ollama/test")
        self.assertIsNone(result["warning"])


if __name__ == "__main__":
    unittest.main()
