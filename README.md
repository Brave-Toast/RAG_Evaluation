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

1. **Select a Benchmark Question**: Choose from 52 pre-loaded HotpotQA multi-hop questions or type a custom query.
2. **Select a Static Perturbation Type**:
   - **Spelling Typo Injection**: Swaps, drops, or substitutes characters based on the Typo Severity slider.
   - **LLM Query Rephrase**: Rephrases the input question while maintaining exact intent.
   - **Conflicting Context**: Injects synthetic counter-factual documents into the vector store.
3. **Customize Modular Pipeline Components**:
   - **Embedding Model**: `HuggingFace / all-MiniLM-L6-v2`.
   - **Search Strategy**: `Similarity` or `MMR (Maximal Marginal Relevance)`.
   - **Top-K Documents**: Set retrieved passage count (1 to 5).
   - **LLM Provider**: Choose `Ollama / llama3`, `Ollama / mistral`, `Gemini 3 Flash`, or `Synthetic Benchmark LLM`.
4. **Execute & Analyze Telemetry**: Click **Run Evaluation Pipeline** to view:
   - Side-by-side comparative execution panels.
   - Granular stage error telemetry bar (Query Shift, Rank Jaccard Overlap, Context Loss, Gen vs GT).
   - Mathematical Error Amplification Factor.
   - Status Classification (`Absorbed` / `Proportional` / `Amplified`).
   - Benchmark history log and Chart.js error degradation tracking.

---

## Troubleshooting & Tips

- **Ollama Offline Notice**:
  If you select `ollama/llama3` or `ollama/mistral` and Ollama is not running on your host machine (`http://localhost:11434`), the backend will log a warning and fall back to the synthetic benchmark generator automatically. To run local LLMs, start Ollama locally using `ollama run llama3`.

- **API Documentation**:
  When the backend is running, interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.
