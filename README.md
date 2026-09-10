# Modular RAG Evaluation Framework & Error Propagation Dashboard

A modular, full-stack evaluation framework built to systematically analyze how perturbations (query spelling errors, LLM query rephrasing, and conflicting context documents) propagate across RAG (Retrieval-Augmented Generation) pipeline stages.

---

## Key Features & Architecture

- **Modular RAG Pipeline Backend**: Built with **FastAPI** ([`backend/app.py`](file:///c:/Users/Mully/OneDrive/Desktop/Uni%20Work/ICT%20Project%20B/RAG_Evaluation/backend/app.py)), **LangChain**, and **ChromaDB** vector store.
- **Dense Vector Embedding Engine**: Powered by `sentence-transformers` (`all-MiniLM-L6-v2`) for accurate dense semantic embeddings.
- **Flexible Vector Search Strategies**: Toggle between **Standard Similarity Search** and **Maximal Marginal Relevance (MMR)** re-ranking to balance document relevance vs diversity.
- **Local & Cloud LLM Integrations**:
  - Local LLMs via **Ollama** (`ollama/llama3`, `ollama/mistral` at `http://localhost:11434/api/generate`).
  - Cloud APIs via **Google Gemini Flash** preview (`gemini-3-flash-preview`).
  - Resilient synthetic benchmark fallback logic.
- **50+ Benchmark Dataset**: Dynamic loading of 52 real multi-hop questions from HotpotQA ([`backend/hotpotqa_dataset.json`](file:///c:/Users/Mully/OneDrive/Desktop/Uni%20Work/ICT%20Project%20B/RAG_Evaluation/backend/hotpotqa_dataset.json)).
- **Granular Stage Telemetry Metrics**:
  1. **Query Embedding Shift**: $\text{CosineSim}(E(Q_{\text{baseline}}), E(Q_{\text{perturbed}}))$
  2. **Retrieval Rank Shift**: Top-K Jaccard document overlap $J(D_{\text{baseline}}, D_{\text{perturbed}}) = \frac{|D_b \cap D_p|}{|D_b \cup D_p|}$
  3. **Context Construction Loss**: $\text{CosineSim}(E(C_{\text{baseline}}), E(C_{\text{perturbed}}))$
  4. **Generation vs. Ground Truth**: Dense similarity comparing baseline and perturbed generated answers against the ground truth.
  5. **Error Amplification Factor**: $\frac{1 - \text{Sim}_{\text{generation}}}{(1 - \text{Sim}_{\text{retrieval}}) + 0.00001}$

---

## Directory Structure

```
RAG_Evaluation/
├── README.md                     # Framework documentation & quickstart guide
├── index.html                    # Frontend Single Page Dashboard (Tailwind CSS & Chart.js)
├── css/
│   └── styles.css                # Extracted visual design system stylesheet
├── js/
│   ├── app.js                    # Main event controller & telemetry orchestrator
│   ├── dataset.js                # Async loader for 50+ HotpotQA benchmark dataset
│   ├── metrics.js                # Metric utilities & status classifiers
│   ├── perturbations.js          # Typo injection, rephrase, & counter-factual context generators
│   ├── ragEngine.js              # Client bridge to FastAPI backend (/api/run-evaluation)
│   ├── storage.js                # Persisted localStorage & CSV/JSON exporter
│   └── uiManager.js              # Chart.js rendering & telemetry card updater
└── backend/
    ├── app.py                    # FastAPI server (ChromaDB, SentenceTransformers, Ollama)
    ├── hotpotqa_dataset.json     # 52 HotpotQA benchmark questions
    └── requirements.txt          # Python dependencies
```

---

## Quickstart Guide

### 1. Prerequisites
- **Python 3.9+** (Python 3.10-3.13 recommended)
- Optional: **Ollama** (if running local models like `llama3` or `mistral`)

### 2. Install Backend Dependencies
Open your terminal in the project directory:

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the FastAPI Backend Server
Launch the server from the `backend/` directory:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

> [!NOTE]
> **Port Conflicts / Permission Error (`WinError 10013` or `WinError 10048`)**:
> If port `8000` is in use by a background process, you can free it by stopping existing Python processes:
> ```powershell
> Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
> ```
> Alternatively, launch uvicorn on port `8001`:
> ```bash
> python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
> ```

### 4. Open the Frontend Dashboard
Once the backend is running, open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
*(Or open `index.html` directly in your browser).*

The dashboard will connect to `http://127.0.0.1:8000/api/health`, dynamically fetch the 52-question HotpotQA dataset, and display the active **FastAPI & ChromaDB / Dense Retriever** status badge!

---

## How to Use the Dashboard

1. Select a benchmark question and configure the embedding model, search strategy, Top-K, and generation provider.
2. Select one or more of the **nine perturbations**: spelling errors, query rephrasing, emoji perturbation, conflicting context, evidence removal, irrelevant passages, fact alteration, passage shuffling, and truncation.
   A single method from either group is sufficient; you do not need to select both a question change and a document change.
3. Choose **Combined** to apply all selected methods to one variant, or **Separate comparisons** to run each method independently against one shared baseline.
4. Adjust the selected methods' controls and optionally enter a random seed. The dashboard shows the number of answer generations before running.
5. Click **Run Evaluation Pipeline**. Progress identifies the shared baseline and each variant. Use **Comparison result** to switch between variants.
6. Inspect **Changes and skipped methods**, the final context supplied to generation, and the expandable retrieved passages before edits. Retrieved-document overlap measures retrieval output; context similarity measures final context after edits.
7. Export history as JSON or CSV. Each variant includes its shared evaluation ID, settings, seed, full result, and generation source. Existing saved history remains readable.

Question rephrasing, spelling errors, emoji perturbation, and conflicting candidate documents run before retrieval. The other five methods run after retrieval. Combined order is fixed: rephrase, spelling, emoji, conflicting context, retrieve, remove evidence, alter facts, truncate, add distractors, shuffle, generate.

**Emoji perturbation:** choose **Insert emojis**, **Encode words and dates**, or **Insert + encode**. Examples include `before → b4️⃣`, `great → gr8️⃣`, and `2026 → 2️⃣0️⃣2️⃣6️⃣`. Adjust the maximum edit count and insertion position. The transformed question and exact replacements are recorded in the comparison result, history, and exports; word encodings can introduce ambiguity.

**Evidence removal:** the bundled dataset has no supporting-document annotations, so results explicitly report the fallback: remove highest-ranked passages. The dataset and stored database are unchanged.

**Synthetic mode:** answers are scripted from the ground truth; use it to verify the workflow, not to measure model resilience. Generation fallback and rule-based rephrasing warnings appear in the results. Seeds reproduce random transformations, not LLM responses.

See [Perturbation experiments](docs/PERTURBATIONS.md) for settings, API examples, interpretation, and tests.

---

## Troubleshooting & Tips

- **Ollama Offline Notice**:
  If you select `ollama/llama3` or `ollama/mistral` and Ollama is not running on your host machine (`http://localhost:11434`), the backend will log a warning and fall back to the synthetic benchmark generator automatically. To run local LLMs, start Ollama locally using `ollama run llama3`.

- **API Documentation**:
  When the backend is running, interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.
