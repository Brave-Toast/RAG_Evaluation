"""Offline tests for the reranking stage: fake cross encoder, no model downloads or ollama needed."""
import asyncio
import copy
import importlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

#same trick as test_evaluation.py, fake the heavy imports so nothing gets loaded
with patch.dict(sys.modules, {
    "sentence_transformers": types.SimpleNamespace(SentenceTransformer=Mock(return_value=None)),
    "langchain_chroma": types.SimpleNamespace(Chroma=Mock()),
    "langchain_core.documents": types.SimpleNamespace(Document=Mock()),
}):
    backend = importlib.import_module("backend.app")


class FakeReranker:
    #gives each doc text a fixed score so the tests know exactly wat order to expect
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return np.array([self.scores.get(text, 0.0) for _, text in pairs])


def doc(title, text):
    return backend.ContextDoc(title=title, text=text)


class RerankDocsTests(unittest.TestCase):
    def setUp(self):
        self.docs = [doc("A", "a text"), doc("B", "b text"), doc("C", "c text")]

    def test_reorders_by_score_and_keeps_top_k(self):
        #logits like ms marco gives (negative n >1), should get sigmoided into 0-1
        fake = FakeReranker({"a text": -2.0, "b text": 5.0, "c text": 1.0})
        with patch.object(backend, "get_reranker", return_value=fake):
            out = backend.rerank_docs("q", self.docs, 2)
        self.assertEqual([d.title for d in out], ["B", "C"])
        self.assertAlmostEqual(out[0].score, 0.9933, places=3)
        self.assertTrue(all(0 <= d.score <= 1 for d in out))

    def test_probability_scores_are_not_sigmoided_twice(self):
        #bge already gives 0-1 probs, they shud come thru as is (the 0.731 bug)
        fake = FakeReranker({"a text": 0.2, "b text": 0.9, "c text": 0.5})
        with patch.object(backend, "get_reranker", return_value=fake):
            out = backend.rerank_docs("q", self.docs, 2)
        self.assertEqual([d.title for d in out], ["B", "C"])
        self.assertEqual([d.score for d in out], [0.9, 0.5])

    def test_scores_doc_text_only_not_title(self):
        #reranker must see the same input as the retriever, no title (test labels live in titles)
        fake = FakeReranker({})
        with patch.object(backend, "get_reranker", return_value=fake):
            backend.rerank_docs("q", self.docs, 2)
        self.assertEqual([text for _, text in fake.calls[0]], ["a text", "b text", "c text"])

    def test_missing_model_raises_instead_of_silently_skipping(self):
        with patch.object(backend, "get_reranker", return_value=None):
            with self.assertRaises(RuntimeError):
                backend.rerank_docs("q", self.docs, 2)


class RetrieveDocsTests(unittest.TestCase):
    def setUp(self):
        self.docs = [doc("A", "a text"), doc("B", "b text"), doc("C", "c text")]
        self.retriever = Mock(side_effect=lambda docs, q, s, k, p: copy.deepcopy(docs[:k]))
        self.req = types.SimpleNamespace(use_reranker=False, search_type="similarity", top_k=2,
                                         embedding_provider="x", rerank_fetch_k=3,
                                         reranker_model=backend.DEFAULT_RERANKER)

    def test_off_is_plain_retrieval_with_no_pool(self):
        with patch.object(backend, "perform_vector_retrieval", self.retriever):
            out, pool = backend.retrieve_docs(self.docs, "q", self.req)
        self.assertEqual(self.retriever.call_args.args[3], 2)  #asked for top_k
        self.assertIsNone(pool)
        self.assertEqual([d.title for d in out], ["A", "B"])

    def test_on_fetches_pool_then_reranks_down_to_top_k(self):
        self.req.use_reranker = True
        fake = FakeReranker({"a text": 0.1, "b text": 0.2, "c text": 0.9})
        with patch.object(backend, "perform_vector_retrieval", self.retriever), \
             patch.object(backend, "get_reranker", return_value=fake):
            out, pool = backend.retrieve_docs(self.docs, "q", self.req)
        self.assertEqual(self.retriever.call_args.args[3], 3)  #asked for fetch_k not top_k
        self.assertEqual(len(pool), 3)
        self.assertEqual([d.title for d in out], ["C", "B"])  #reranker promoted C from last to first

    def test_pool_is_never_smaller_than_top_k(self):
        self.req.use_reranker = True
        self.req.rerank_fetch_k = 1  #smaller than top_k=2
        with patch.object(backend, "perform_vector_retrieval", self.retriever), \
             patch.object(backend, "get_reranker", return_value=FakeReranker({})):
            backend.retrieve_docs(self.docs, "q", self.req)
        self.assertEqual(self.retriever.call_args.args[3], 2)


class RerankApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = pathlib.Path(self.temp.name) / "stages.jsonl"
        self.retriever = Mock(side_effect=lambda docs, q, s, k, p: copy.deepcopy(docs[:k]))
        self.fake = FakeReranker({"Started in 1844.": 0.2, "Started in 1989.": 0.9})

        def embed(text, provider):
            vector = np.ones(16, dtype=np.float32)
            for char in text:
                vector[ord(char) % 16] += 1
            return vector

        for patcher in [patch.object(backend, "STAGE_LOG_PATH", self.log),
                        patch.object(backend, "perform_vector_retrieval", self.retriever),
                        patch.object(backend, "compute_dense_embedding", side_effect=embed),
                        patch.object(backend, "load_hotpotqa_dataset", return_value=[]),
                        patch.object(backend, "get_reranker", return_value=self.fake)]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(backend.app)
        self.payload = {
            "question_id": "one", "baseline_query": "Which magazine started first?",
            "ground_truth": "First", "llm_model": "synthetic-evaluator", "seed": 42,
            "contexts": [{"title": "First", "text": "Started in 1844."},
                         {"title": "Second", "text": "Started in 1989."}],
            "perturbation_types": ["spelling"], "top_k": 1, "rerank_fetch_k": 2,
        }

    def test_unknown_reranker_is_rejected(self):
        res = self.client.post("/api/run-evaluation", json={**self.payload, "use_reranker": True, "reranker_model": "not/a-model"})
        self.assertEqual(res.status_code, 422)
        self.retriever.assert_not_called()

    def test_reranker_list_endpoint(self):
        data = self.client.get("/api/rerankers").json()
        self.assertEqual(data["default"], backend.DEFAULT_RERANKER)
        self.assertEqual({r["name"] for r in data["rerankers"]}, set(backend.SUPPORTED_RERANKERS))

    def test_on_reranks_baseline_and_perturbed_and_logs_the_stage(self):
        res = self.client.post("/api/run-evaluation", json={**self.payload, "use_reranker": True})
        self.assertEqual(res.status_code, 200, res.text)
        result = res.json()
        #reranker scored Second higher so it shud win in both runs, fair comparison
        self.assertEqual([d["title"] for d in result["baseline_contexts"]], ["Second"])
        self.assertEqual([d["title"] for d in result["perturbed_retrieved_contexts"]], ["Second"])
        self.assertTrue(all(call.args[3] == 2 for call in self.retriever.call_args_list))  #every retrieval grabbed fetch_k
        stage = json.loads(self.log.read_text(encoding="utf-8").splitlines()[-1])["rerank_stage"]
        self.assertTrue(stage["enabled"])
        self.assertEqual(stage["model"], backend.DEFAULT_RERANKER)
        self.assertEqual(len(stage["baseline_pool"]), 2)
        self.assertEqual(stage["baseline_kept"], ["Second"])

    def test_off_logs_disabled_and_never_touches_reranker(self):
        res = self.client.post("/api/run-evaluation", json=self.payload)
        self.assertEqual(res.status_code, 200, res.text)
        stage = json.loads(self.log.read_text(encoding="utf-8").splitlines()[-1])["rerank_stage"]
        self.assertFalse(stage["enabled"])
        self.assertIsNone(stage["baseline_pool"])
        self.assertEqual(self.fake.calls, [])


class RerankingLocalisationTests(unittest.TestCase):
    #swap test shud blame RERANKING only wen the retrivers own picks wud of given the right answer
    def setUp(self):
        self.gold = doc("Gold", "Paris is the capital of France.")
        self.bad = doc("Fake", "Lyon is a big city.")
        self.req = types.SimpleNamespace(ground_truth="Paris", llm_model="ollama/test", gemini_api_key=None, top_k=1)

        async def fake_llm(query, contexts, **kwargs):
            #answers right only if the gold doc is in the context
            return ("Paris" if any(c.title == "Gold" for c in contexts) else "Lyon"), None
        patcher = patch.object(backend, "generate_llm_answer", new=fake_llm)
        patcher.start()
        self.addCleanup(patcher.stop)

    def localise(self, pool):
        #clean run right, perturbed run wrong because the reranker kept the fake doc
        return asyncio.run(backend.localise_failure(
            self.req, "q", [self.gold], [self.bad], "Paris", "Lyon", False, ["Gold"], pool, pool))

    def test_blames_reranking_when_retrievers_own_pick_was_right(self):
        self.assertEqual(self.localise([self.gold, self.bad])["failed_at"], "RERANKING")

    def test_blames_retrieval_when_retriever_was_already_wrong(self):
        self.assertEqual(self.localise([self.bad, self.gold])["failed_at"], "RETRIEVAL")

    def test_reranker_off_behaves_like_before(self):
        self.assertEqual(self.localise(None)["failed_at"], "RETRIEVAL")


if __name__ == "__main__":
    unittest.main()