# RAG Perturbation-Propagation Evaluation Framework

An evaluation framework and dashboard for studying how **perturbations** (noisy queries, damaged or poisoned documents) propagate through a Retrieval-Augmented Generation (RAG) pipeline and degrade answer quality, and **which pipeline stage is responsible** when an answer goes wrong.

Built for ICT30018 ICT Project B (Swinburne University of Technology). Dataset: HotpotQA (distractor setting).

---

## What it does

1. Runs a clean **baseline** RAG pipeline for a question.
2. Applies one or more **perturbations** (12 available) before or after retrieval.
3. Optionally applies **repair methods** and/or a **cross-encoder reranker**.
4. Compares baseline and perturbed runs with similarity metrics, token F1 and an **LLM-as-judge**.
5. Runs a **swap test** that puts clean stages back one at a time to localise the failing stage: QUERY, RETRIEVAL, RERANKING (only when the reranker is on), CONTEXT or GENERATION.

```
User Query -> Embedding -> Retrieval (ChromaDB) -> [Reranker] -> Context Construction -> Generation (LLM) -> Evaluation
   ^                                  ^                                  ^
 query perturbations           candidate-document perturbations     document perturbations
 + query repairs               (conflict, metadata_injection)       + context repairs
```

---

## Key features

- **Modular backend:** FastAPI (`backend/app.py`), LangChain, ChromaDB.
- **Embeddings:** `all-mpnet-base-v2` (default), `all-MiniLM-L6-v2`, Ollama `nomic-embed-text`, OpenAI `text-embedding-3-small`, and an `ensemble/mpnet+nomic` option.
- **Search:** standard similarity or MMR ("MMR (Diverse)").
- **Reranker stage (optional):** the retriever fetches 10 candidates (`rerank_fetch_k`) and a cross-encoder keeps the best `top_k`. Models: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base` (recommended, default) and `BAAI/bge-reranker-v2-m3`. Models load lazily and are cached.
- **LLMs:** local Ollama (`qwen3.5:9b` default, `llama3`, `mistral`), Gemini (`gemini-3-flash-preview`), or `synthetic-evaluator` (offline, scripted from ground truth; for checking the workflow only).
- **LLM-as-judge:** scores answer correctness, faithfulness and context relevance (`backend/llm_judge.py`). Reported alongside token F1, since the two disagree on some answers.
- **Failure diagnosis:** `backend/isolation_metrics.py` classifies runs as SUCCESS, RETRIEVAL_FAILURE, GENERATION_FAILURE or PARAMETRIC_LEAKAGE, and the swap test adds stage-level blame, including RERANKING.
- **Repair methods:** registered in `backend/repairs.py` and listed by `/api/repairs`.
- **Dataset:** 80 HotpotQA multi-hop questions in `backend/hotpotqa_dataset.json`.
- **Telemetry:** query embedding shift, retrieval overlap (Jaccard), final-context similarity, generation similarity, gold-context recall, answer F1, judge scores and an error amplification factor.
- **Reproducibility:** seeded perturbations (seeds reproduce random edits, not LLM responses); every run is logged to `backend/stage_run_logs.jsonl`.

---

## Perturbations (12)

| Method | Stage | What it does |
|---|---|---|
| `spelling` | Before retrieval | Random character swaps, deletions and substitutions in the question (`typo_rate`). |
| `rephrase` | Before retrieval | LLM rephrase of the question, with a labelled rule-based fallback. |
| `emoji` | Before retrieval | Insert emojis or encode words/dates as emoji (`emoji_style`, `emoji_count`, `emoji_position`). |
| `negation_flip` | Before retrieval | Flips the question's polarity: removes an existing negation, or inserts one after the first auxiliary verb. |
| `entity_substitution` | Before retrieval | Swaps a named entity (a known document title) in the question for an unrelated one from another benchmark question. |
| `conflict` | Before retrieval | Adds a contradictory candidate document (it embeds the query, so it is retrieved very easily). |
| `metadata_injection` | Before retrieval | Adds an off-topic passage with query-derived topic keywords prepended (`metadata_term_count`), so it ranks highly without containing the answer. |
| `remove_evidence` | After retrieval | Removes retrieved passages (`remove_count`). |
| `alter_facts` | After retrieval | Replaces sampled integers in passages (`fact_count`). |
| `truncate` | After retrieval | Keeps only a prefix of each passage (`retain_percent`). |
| `irrelevant` | After retrieval | Appends passages from other questions (`irrelevant_count`). |
| `shuffle` | After retrieval | Reorders passages without editing text. |

Use **Combined** to apply all selected methods to one variant, or **Separate comparisons** to run each independently against a shared baseline. Fixed combined order: rephrase, spelling, emoji, negation flip, entity substitution, metadata injection, conflict, retrieve, remove evidence, alter facts, truncate, irrelevant, shuffle, generate.

See [docs/PERTURBATIONS.md](docs/PERTURBATIONS.md) for settings and API examples.

## Repair methods

Registered in `backend/repairs.py`: `fix_spelling`, `contradiction_filter`, `contradiction_filter_text_only`, `fix_shuffle`, `fix_truncate`, `fix_alter_facts`, `fix_alter_facts_drop` (plus a test-only `noop`). Repairs run after retrieval (spelling repair runs on the query). See `RS-44_RS-46_findings_summary.md`, `RS-62_report_section.md` and `results/comparison_table.md` for what they recover.

---

## Quickstart

### 1. Prerequisites
- Python 3.10-3.13
- [Ollama](https://ollama.com) with `qwen3.5:9b` pulled (optional; missing models are auto-pulled, and the synthetic evaluator works without Ollama)
- A GPU is recommended for local LLM runs

### 2. Install
```bash
cd backend
pip install -r requirements.txt
```
Optional `.env` file in `backend/` for cloud providers:
```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

### 3. Run the server
```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Then open `http://127.0.0.1:8000/`. API docs: `http://127.0.0.1:8000/docs`.

If port 8000 is busy, use `--port 8001`.

### 4. Use the dashboard
1. Select a question and configure the embedding model, search type, Top-K, reranker and LLM.
2. Add perturbations (and optionally repairs), choose Combined or Separate, and optionally set a seed.
3. Click **Run Evaluation Pipeline**. Inspect the Root Attribution card, telemetry, retrieved and final context, and Benchmark History.
4. Export history as JSON or CSV.

---

## Tests
```bash
python -m pytest tests -v
```
Covers perturbations, repairs, isolation metrics, the LLM judge, attribution scenarios and the reranker (`tests/test_reranker.py`: 14 offline tests with a fake cross-encoder).

## Experiments and scripts (run from `backend/` with the server running)

| Script | Purpose |
|---|---|
| `batch_reranker.py` | 80 questions x 6 pre-retrieval perturbations, reranker off vs on (recall only). Results in `results/reranker_batch*.csv`. |
| `pipeline_robustness_benchmark.py` | Full pipeline (BGE reranker, LLM answers and judge, all 12 perturbations, 40 questions, seed 42) at a chosen `--top-k`. Results in `results/robustness_benchmark/`. |
| `run_comparison.py` / `build_comparison_table.py` | Perturbation x repair comparison table. |
| `benchmark_repair_effectiveness.py` | Repair effectiveness across questions. |

---

## Directory structure
```
RAG_Evaluation/
├── README.md
├── index.html            # Dashboard
├── css/ js/              # Dashboard code (app.js, uiManager.js, llmJudgeUI.js, ...)
├── backend/
│   ├── app.py            # FastAPI server, pipeline, reranker, swap test
│   ├── perturbations.py  # The 12 perturbation methods
│   ├── repairs.py        # Repair registry and methods
│   ├── isolation_metrics.py, llm_judge.py
│   ├── ingest.py, hotpotqa_dataset.json, requirements.txt
│   └── batch / benchmark scripts
├── tests/                # Unit and scenario tests
├── docs/PERTURBATIONS.md # Perturbation settings and API examples
└── results/              # Experiment outputs (CSV / markdown)
```

## Limitations
- Results are from one dataset (HotpotQA), a small question set (40 for the full benchmark, 80 for the reranker batch) and mostly one local model (`qwen3.5:9b`).
- The reranker batch measures context recall with synthetic answers; the `conflict` perturbation embeds the query, so it is easier to retrieve than a real adversarial document.
- The evidence-removal perturbation falls back to removing top-ranked passages because the bundled dataset has no supporting-document annotations.
- Seeds reproduce random transformations but not LLM responses.
