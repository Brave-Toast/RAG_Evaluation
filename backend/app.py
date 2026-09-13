"""
FastAPI Backend Server for True RAG Evaluation, Modular Vector Retrieval (ChromaDB),
Local LLM Integration (Ollama), and Granular Dense Vector Telemetry.
"""

import os
import json
import random
import math
import pathlib
import asyncio
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import httpx

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

if __package__:
    from .perturbations import (Method, PRE_RETRIEVAL, PerturbationSettings, ordered_methods,
                                method_rng, change_record, distractor_pool, apply_context_methods, perturb_query_with_emojis)
    from .isolation_metrics import (calculate_token_f1, calculate_exact_match,
                                    calculate_context_recall, calculate_context_precision,
                                    diagnose_failure)
else:
    from perturbations import (Method, PRE_RETRIEVAL, PerturbationSettings, ordered_methods,
                               method_rng, change_record, distractor_pool, apply_context_methods, perturb_query_with_emojis)
    from isolation_metrics import (calculate_token_f1, calculate_exact_match,
                                  calculate_context_recall, calculate_context_precision,
                                  diagnose_failure)

app = FastAPI(
    title="Modular RAG Evaluation & Perturbation Backend API",
    description="Provides dense vector embeddings, ChromaDB retrieval (Similarity/MMR), local Ollama LLM integration, and granular pipeline error telemetry.",
    version="2.1.0"
)

# Enable CORS for local dashboard development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = pathlib.Path(__file__).parent.parent


from dotenv import load_dotenv
load_dotenv()
# OpenAI API key is loaded from the .env file. Every team mmeber if running locally, needs ot create thier own .env file in the smae direcotry as app.py and set the key as OPENAI_API_KEY=sk-proj-yourrealkeyhere 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

@app.get("/")
def read_root():
    """Serves the frontend dashboard SPA."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "RAG Evaluation Framework API is running."}

# Mount static files (css, js, images)
if FRONTEND_DIR.exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")

# SentenceTransformer model registry & loader
EMBEDDING_MODELS: Dict[str, Any] = {}

def get_sentence_transformer(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Loads or retrieves cached SentenceTransformer model instance."""
    normalized_name = model_name.replace("huggingface/", "").strip()
    if normalized_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[normalized_name]
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading SentenceTransformer model '{normalized_name}'...")
        model = SentenceTransformer(normalized_name)
        EMBEDDING_MODELS[normalized_name] = model
        print(f"SentenceTransformer '{normalized_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Notice: Could not load SentenceTransformer '{normalized_name}': {e}. Fallback will be used.")
        return None

# Preload default embedding model on startup
try:
    get_sentence_transformer("sentence-transformers/all-mpnet-base-v2")
except Exception:
    pass

# ChromaDB / LangChain integration availability
CHROMA_AVAILABLE = False
try:
    import uuid
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        CHROMA_AVAILABLE = True
    except ImportError:
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        CHROMA_AVAILABLE = True
    print("ChromaDB & LangChain vectorstore available.")
except Exception as e:
    print(f"ChromaDB/LangChain import notice: {e}. Custom dense distance vector retrieval will be used.")


class ContextDoc(BaseModel):
    title: str
    text: str
    score: Optional[float] = None


class EvaluationRequest(BaseModel):
    question_id: Optional[str] = None
    custom_query: Optional[str] = None
    baseline_query: str
    ground_truth: str
    contexts: List[ContextDoc]
    gold_titles: Optional[List[str]] = Field(default_factory=list)
    perturbation_type: Method = "spelling"  # Legacy single-method requests remain supported.
    perturbation_types: Optional[List[Method]] = Field(default=None, min_length=1, max_length=9)
    mode: Literal["combined", "separate"] = "combined"
    settings: PerturbationSettings = Field(default_factory=PerturbationSettings)
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)
    supporting_titles: Optional[List[str]] = None
    typo_rate: float = Field(default=0.25, ge=0, le=1)
    embedding_provider: str = "sentence-transformers/all-mpnet-base-v2"
    search_type: str = "similarity"  # "similarity" or "mmr"
    top_k: int = Field(default=2, ge=1, le=5)
    llm_model: str = "ollama/qwen3.5:9b"  # "ollama/qwen3.5:9b", "ollama/llama3", "ollama/mistral", "gemini-3-flash-preview", "synthetic-evaluator"
    gemini_api_key: Optional[str] = None


class IngestRequest(BaseModel):
    num_rows: int = 50
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 200
    chunk_overlap: int = 50


class MetricTelemetry(BaseModel):
    query_embedding_shift: float
    retrieval_rank_shift_jaccard: float
    context_construction_loss: float
    generation_vs_gt_baseline: float
    generation_vs_gt_perturbed: float
    retrieval_sim: float
    generation_sim: float
    amplification_factor: float
    status_category: str
    status_text: str
    status_desc: str
    context_recall_baseline: float
    context_recall_perturbed: float
    answer_f1_baseline: float
    answer_f1_perturbed: float
    failure_attribution: str
    failure_root_cause: str
    context_precision_baseline: Optional[float] = 1.0
    context_precision_perturbed: Optional[float] = 1.0


class VariantResult(BaseModel):
    baseline_query: str
    perturbed_query: str
    baseline_contexts: List[ContextDoc]
    perturbed_contexts: List[ContextDoc]
    baseline_answer: str
    perturbed_answer: str
    ground_truth: str
    metrics: MetricTelemetry
    warning: Optional[str] = None
    variant_id: str
    methods: List[Method]
    application_order: List[Method]
    changes: List[Dict[str, Any]]
    baseline_retrieved_contexts: List[ContextDoc]
    perturbed_retrieved_contexts: List[ContextDoc]
    baseline_generation_source: str
    perturbed_generation_source: str


class EvaluationResponse(VariantResult):
    # Top-level comparison fields mirror the first variant for existing clients.
    evaluation_id: str
    mode: str
    selected_methods: List[Method]
    settings: Dict[str, Any]
    seed: int
    variants: List[VariantResult]


DATASET_PATH = pathlib.Path(__file__).parent / "hotpotqa_dataset.json"
STAGE_LOG_PATH = pathlib.Path(__file__).parent / "stage_run_logs.jsonl"


def log_stage_run(
    req: "EvaluationRequest",
    baseline_query: str,
    perturbed_query: str,
    baseline_contexts: List[ContextDoc],
    perturbed_contexts: List[ContextDoc],
    baseline_answer: str,
    perturbed_answer: str,
    metrics: "MetricTelemetry",
    warning: Optional[str] = None,
    experiment: Optional[Dict[str, Any]] = None,
    gold_titles: Optional[List[str]] = None
) -> None:
    """
    Appends a structured, stage-level record of this evaluation run to a JSONL log file.
    Captures query, retrieval, and generation stage outputs separately so later analysis
    can trace which pipeline stage a failure originated from (see RS-41).
    """
    import datetime
    import uuid as uuid_module

    resolved_gold = gold_titles if gold_titles is not None else (req.gold_titles or [])
    record = {
        "run_id": uuid_module.uuid4().hex,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "question_id": req.question_id,
        "perturbation_type": req.perturbation_type,
        "typo_rate": req.typo_rate,
        "embedding_provider": req.embedding_provider,
        "search_type": req.search_type,
        "top_k": req.top_k,
        "llm_model": req.llm_model,
        "gold_titles": resolved_gold,
        "context_recall": metrics.context_recall_perturbed,
        "failure_attribution": metrics.failure_attribution,
        "failure_root_cause": metrics.failure_root_cause,
        "query_stage": {
            "original_query": baseline_query,
            "perturbed_query": perturbed_query
        },
        "retrieval_stage": {
            "baseline_contexts": [
                {"title": c.title, "score": c.score} for c in baseline_contexts
            ],
            "perturbed_contexts": [
                {"title": c.title, "score": c.score} for c in perturbed_contexts
            ]
        },
        "generation_stage": {
            "baseline_answer": baseline_answer,
            "perturbed_answer": perturbed_answer
        },
        "metrics": metrics.model_dump(),
        "warning": warning
    }
    if experiment:
        record.update(experiment)

    try:
        with open(STAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Notice: could not write stage log entry: {e}")

def load_hotpotqa_dataset() -> List[Dict[str, Any]]:
    """Loads benchmark questions from hotpotqa_dataset.json."""
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

Ollama_Confirmed_Models: set=set()
def ensure_ollama_model_pulled(model_name:str)->bool:
    #checking if ollama model is available locally
    if model_name in Ollama_Confirmed_Models:
        return True
    try:
        with httpx.Client(timeout=5.0) as client:
            # quickly trying a embedidng call for checking if model exists
            res=client.post("http://localhost:11434/api/embeddings", json={"model": model_name, "prompt": "ping"})
            if res.status_code==200:
                Ollama_Confirmed_Models.add(model_name)
                return True
    except Exception:
        pass
    # if  not found then pull the model 
    print(f"Model '{model_name}' not found locally, pulling from ollama")
    try:
        with httpx.Client(timeout=300.0) as client:
            res=client.post("http://localhost:11434/api/pull", json={"name": model_name, "stream": False})
            if res.status_code==200:
                print(f"Pulled '{model_name}' successfully.") 
                Ollama_Confirmed_Models.add(model_name)
                return True
            else:
                 print(f"Pull failed for '{model_name}': HTTP {res.status_code}")
    except Exception as e:
        print(f"Pull error for '{model_name}': {e}")
    return False
          
def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Converts an embedding to float32 and normalizes it to unit length."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_dense_embedding(text: str, provider: str = "sentence-transformers/all-mpnet-base-v2") -> np.ndarray:
    """Computes a dense vector embedding using Ollama, SentenceTransformers, or fallback encoder."""
    if not text or not text.strip():
        return np.zeros((768 if "mpnet" in provider else 384,), dtype=np.float32)
    
    # 1. Ollama local embeddings
    if provider and provider.startswith("ollama/"):
        model_name = provider.replace("ollama/", "")
        if ensure_ollama_model_pulled(model_name):
            try:
                with httpx.Client(timeout=300.0) as client:
                    res = client.post("http://localhost:11434/api/embeddings", json={"model": model_name, "prompt": text})
                    if res.status_code == 200:
                        emb = res.json().get("embedding")
                        if emb:
                            return normalize_vector(emb)
            except Exception as e:
                print(f"Ollama embeddings error for '{model_name}': {e}")
        else:
            print(f"Notice: could not use ollama model '{model_name}', falling back.")
# 1b. OpenAI embeddings
    if provider and provider.startswith("openai/"):
        model_name = provider.replace("openai/", "")
        if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
            try:
                with httpx.Client(timeout=10.0) as client:
                    res = client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={"model": model_name, "input": text}
                    )
                    if res.status_code == 200:
                        emb = res.json().get("data", [{}])[0].get("embedding")
                        if emb:
                            return normalize_vector(emb)
                    else:
                        print(f"OpenAI embeddings HTTP {res.status_code}: {res.text[:200]}")
            except Exception as e:
                print(f"OpenAI embeddings error: {e}")
        else:
            print("Notice: OPENAI_API_KEY not set, falling back.")

            
    # 2. SentenceTransformer local embeddings
    model_key = provider.replace("huggingface/", "")
    st_model = get_sentence_transformer(model_key)
    if st_model is None and "all-mpnet-base-v2" not in model_key:
        st_model = get_sentence_transformer("all-MiniLM-L6-v2")
    if st_model is None and "all-mpnet-base-v2" in model_key:
        st_model = get_sentence_transformer("sentence-transformers/all-mpnet-base-v2")
        
    if st_model is not None:
        try:
            emb = st_model.encode(text)
            return normalize_vector(emb)
        except Exception as e:
            print(f"SentenceTransformer encode notice: {e}")

    # 3. Fallback hash-based pseudo-embedding (deterministic vector generation)
    dim = 768 if "mpnet" in provider else 384
    vec = np.zeros((dim,), dtype=np.float32)
    words = text.lower().split()
    for w in words:
        idx = hash(w) % dim
        vec[idx] += 1.0
    return normalize_vector(vec)


def compute_dense_cosine_similarity(text1: str, text2: str, provider: str = "sentence-transformers/all-mpnet-base-v2") -> float:
    """Calculates CosineSim(E(text1), E(text2))."""
    if not text1 or not text2:
        return 0.0
    if text1.strip() == text2.strip():
        return 1.0
    vec1 = compute_dense_embedding(text1, provider)
    vec2 = compute_dense_embedding(text2, provider)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return float(np.clip(sim, 0.0, 1.0))


def compute_jaccard_overlap(docs1: List[ContextDoc], docs2: List[ContextDoc]) -> float:
    """Calculates Top-K Jaccard overlap between retrieved document sets."""
    set1 = set(d.title for d in docs1)
    set2 = set(d.title for d in docs2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0
    return float(len(intersection) / len(union))


def perform_vector_retrieval(
    docs: List[ContextDoc], 
    query: str, 
    search_type: str = "similarity", 
    top_k: int = 2,
    embedding_provider: str = "sentence-transformers/all-mpnet-base-v2"
) -> List[ContextDoc]:
    """
    Executes True Vector Retrieval using ChromaDB and LangChain embeddings,
    supporting similarity_search_with_relevance_scores and Maximal Marginal Relevance (MMR).
    Returns retrieved ContextDoc items with their calculated relevance/confidence scores.
    """
    if not docs:
        return []
    
    k_val = min(top_k, len(docs))
    
    if CHROMA_AVAILABLE:
        try:
            class CustomDenseEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [compute_dense_embedding(t, embedding_provider).tolist() for t in texts]
                def embed_query(self, text: str) -> List[float]:
                    return compute_dense_embedding(text, embedding_provider).tolist()
            
            lc_docs = [Document(page_content=d.text, metadata={"title": d.title, "index": idx}) for idx, d in enumerate(docs)]

            # Give this call its own isolated collection so switching embedding
            # models between requests can't collide with a previous dimension.
            # Without a unique collection_name, Chroma.from_documents() reuses
            # a shared default in-memory collection across every call.
            unique_collection_name = f"eval_{uuid.uuid4().hex}"
            db = Chroma.from_documents(
                lc_docs,
                CustomDenseEmbeddings(),
                collection_name=unique_collection_name
            )
            
            result = []
            if search_type.lower() == "mmr":
                retrieved_lc = db.max_marginal_relevance_search(query, k=k_val, fetch_k=min(len(docs), 10))
                for doc in retrieved_lc:
                    title = doc.metadata.get("title", "Untitled")
                    sim_score = compute_dense_cosine_similarity(query, f"{title} {doc.page_content}", provider=embedding_provider)
                    result.append(ContextDoc(title=title, text=doc.page_content, score=round(float(sim_score), 4)))
            else:
                # Use similarity_search_with_relevance_scores
                scored_lc = db.similarity_search_with_relevance_scores(query, k=k_val)
                for doc, score in scored_lc:
                    title = doc.metadata.get("title", "Untitled")
                    score_val = float(score) if score is not None else compute_dense_cosine_similarity(query, f"{title} {doc.page_content}", provider=embedding_provider)
                    # Normalize relevance score within [0, 1] range
                    score_val = max(0.0, min(1.0, score_val))
                    result.append(ContextDoc(title=title, text=doc.page_content, score=round(score_val, 4)))

            # Clean up this temporary collection now that we've read what we need
            try:
                db.delete_collection()
            except Exception as cleanup_err:
                print(f"Notice: could not clean up temp collection '{unique_collection_name}': {cleanup_err}")
                    
            if result:
                return result
        except Exception as e:
            print(f"ChromaDB retrieval fallback notice: {e}")
    
    # Dense vector distance fallback implementation
    scored = []
    for d in docs:
        sim = compute_dense_cosine_similarity(query, f"{d.title} {d.text}", provider=embedding_provider)
        scored.append({"doc": d, "score": sim, "text": d.text})

    if search_type.lower() == "mmr":
        selected = []
        unselected = list(scored)
        lambd = 0.7
        while len(selected) < k_val and unselected:
            best_idx = 0
            best_mmr = -1e9
            for i, cand in enumerate(unselected):
                rel = cand["score"]
                max_redundancy = 0.0
                for sel in selected:
                    sim = compute_dense_cosine_similarity(cand["text"], sel["text"], provider=embedding_provider)
                    if sim > max_redundancy:
                        max_redundancy = sim
                mmr_val = (lambd * rel) - ((1.0 - lambd) * max_redundancy)
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = i
            selected.append(unselected.pop(best_idx))
        return [
            ContextDoc(
                title=item["doc"].title,
                text=item["doc"].text,
                score=round(float(item["score"]), 4)
            ) for item in selected
        ]
    else:
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [
            ContextDoc(
                title=item["doc"].title,
                text=item["doc"].text,
                score=round(float(item["score"]), 4)
            ) for item in scored[:k_val]
        ]


async def generate_llm_answer(
    query: str, 
    contexts: List[ContextDoc], 
    model_choice: str, 
    ground_truth: str, 
    gemini_api_key: Optional[str] = None, 
    is_perturbed: bool = False,
    perturbation_type: str = "spelling"
) -> tuple[str, Optional[str]]:
    """
    Generates an LLM response using local Ollama (qwen3.5:9b, llama3, mistral),
    cloud Gemini API, or benchmark synthetic fallback with robust error handling.
    """
    ctx_text = "\n".join([f"[{c.title}]: {c.text}" for c in contexts])
    
    if not is_perturbed:
        system_prompt = "You are a precise QA assistant. Answer concisely based strictly on the provided pristine baseline context."
        prompt = f"Baseline Context:\n{ctx_text}\n\nQuestion: {query}\nAnswer concisely based strictly on baseline context:"
    else:
        system_prompt = "You are a QA assistant evaluating perturbed RAG context. Answer the question based on context. If context contains conflicting claims, state the conflict clearly."
        prompt = f"Perturbed Context:\n{ctx_text}\n\nQuestion: {query}\nAnswer concisely based on context:"

    warning_msg = None

    # 1. Ollama Local LLM Integration (Supporting qwen3.5:9b, llama3, mistral)
    if model_choice.startswith("ollama/") or model_choice in ["qwen3.5:9b", "llama3", "mistral", "qwen2.5:7b"]:
        model_name = model_choice.replace("ollama/", "")
        ollama_url = "http://localhost:11434/api/generate"
        if ensure_ollama_model_pulled(model_name):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    res = await client.post(
                        ollama_url,
                        json={
                            "model": model_name,
                            "prompt": f"{system_prompt}\n\n{prompt}",
                            "stream": False,
                            "options": {
                                "num_predict": 8000
                            }
                        }
                    )
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("response", "").strip()
                        if answer:
                            return answer, None
                        else:
                            warning_msg = f"Ollama returned empty response for '{model_name}'. Raw data: {data}"
                            print(warning_msg)
                    else:
                        warning_msg = f"Ollama HTTP {res.status_code} for '{model_name}'. Used synthetic fallback."
                        print(warning_msg)
            except httpx.ConnectError:
                warning_msg = f"Ollama service offline (could not reach localhost:11434 for model '{model_name}'). Used synthetic fallback."
                print(warning_msg)
            except httpx.TimeoutException:
                warning_msg = f"Ollama request timed out for model '{model_name}'. Used synthetic fallback."
                print(warning_msg)
            except Exception as e:
                warning_msg = f"Ollama local error for '{model_name}': {str(e)}. Used synthetic fallback."
                print(warning_msg)
        else:
            warning_msg = f"Could not pull ollama model '{model_name}'. Used synthetic fallback."
            print(warning_msg)

    # 2. Gemini Cloud API Integration
    if model_choice == "gemini-3-flash-preview" or (gemini_api_key and model_choice != "synthetic-evaluator" and not model_choice.startswith("ollama/")):
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if api_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "systemInstruction": {"parts": [{"text": system_prompt}]}
                    }
                    res = await client.post(url, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                        if text:
                            return text.strip(), None
            except Exception as e:
                print(f"Gemini API call warning: {e}")

    # Mark scripted answers and provider failures in experiment results.
    warning_msg = warning_msg or ("Synthetic benchmark output (scripted from ground truth)." if model_choice == "synthetic-evaluator" else "Generation unavailable or empty response. Used synthetic fallback.")
    # 3. Synthetic Benchmark Fallback (Clean & Pure Data Integrity)
    if not is_perturbed:
        return ground_truth, warning_msg
    else:
        if perturbation_type == "conflict" or any("Synthetic" in c.title for c in contexts):
            return f"Conflicting sources refute that '{ground_truth}' is accurate.", warning_msg
        else:
            return ground_truth, warning_msg
async def generate_query_rephrase(query: str, model_choice: str, gemini_api_key: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Generates a semantic rephrase of the input query for the perturbed test pipeline.
    Uses local Ollama (including qwen3.5:9b), cloud Gemini API, or high-quality rule-based linguistic rephraser.
    """
    rephrase_system_prompt = "You are a linguistic expert assistant. Rephrase the input question concisely while preserving its exact semantic intent. Return ONLY the rephrased question text."
    prompt = f"Rephrase this question concisely while maintaining exact meaning: \"{query}\""
    warning_msg = None

    # 1. Ollama Local LLM
    if model_choice.startswith("ollama/") or model_choice in ["qwen3.5:9b", "llama3", "mistral"]:
        model_name = model_choice.replace("ollama/", "")
        ollama_url = "http://localhost:11434/api/generate"
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                res = await client.post(
                    ollama_url,
                    json={
                        "model": model_name,
                        "prompt": f"{rephrase_system_prompt}\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "num_predict": 8000
                        }
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    rephrased = data.get("response", "").strip().strip('"')
                    if rephrased and rephrased != query:
                        return rephrased, None
        except Exception:
            warning_msg = f"Ollama local service unreachable for query rephrasing."

    # 2. Gemini Cloud API
    if model_choice == "gemini-3-flash-preview" or (gemini_api_key and model_choice != "synthetic-evaluator" and not model_choice.startswith("ollama/")):
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if api_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "systemInstruction": {"parts": [{"text": rephrase_system_prompt}]}
                    }
                    res = await client.post(url, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                        if text:
                            return text.strip().strip('"'), None
            except Exception as e:
                print(f"Gemini rephrase API call warning: {e}")

    # 3. High-Quality Rule-Based Linguistic Rephraser (Fallback)
    q_clean = query.strip()
    if q_clean.endswith("?"):
        q_clean = q_clean[:-1]

    if q_clean.lower().startswith("which"):
        rephrased = f"Can you specify {q_clean.lower()}?"
    elif q_clean.lower().startswith("what"):
        rephrased = f"Could you identify {q_clean.lower()}?"
    elif q_clean.lower().startswith("were") or q_clean.lower().startswith("was") or q_clean.lower().startswith("is") or q_clean.lower().startswith("are"):
        rephrased = f"Is it true that {q_clean.lower()}?"
    elif q_clean.lower().startswith("who"):
        rephrased = f"Can you tell me {q_clean.lower()}?"
    else:
        rephrased = f"Could you provide details regarding: {q_clean.lower()}?"
        
    return rephrased, warning_msg or "Rule-based rephrase fallback used; semantic equivalence is not guaranteed."


def inject_spelling_typos(text: str, error_rate: float = 0.25, rng=None) -> str:
    """Injects stochastic spelling perturbations into input query string."""
    rng = rng or random
    if error_rate == 0:
        return text
    if not text:
        return ""
    words = text.split()
    result = []
    chars = "abcdefghijklmnopqrstuvwxyz"
    for word in words:
        if len(word) > 3 and rng.random() < error_rate:
            op = rng.choice(["swap", "drop", "sub"])
            idx = rng.randint(1, len(word) - 2)
            if op == "swap":
                arr = list(word)
                arr[idx], arr[idx+1] = arr[idx+1], arr[idx]
                result.append("".join(arr))
            elif op == "drop":
                result.append(word[:idx] + word[idx+1:])
            else:
                sub_char = rng.choice(chars)
                result.append(word[:idx] + sub_char + word[idx+1:])
        else:
            result.append(word)
    return " ".join(result)


def inject_conflicting_context(contexts: List[ContextDoc], query: str, ground_truth: str) -> List[ContextDoc]:
    """Injects a high-priority synthetic contradictory context passage."""
    conflict_doc = ContextDoc(
        title="Synthetic_Conflicting_Record",
        text=f"NOTICE: Regarding the query '{query}', conflicting modern sources refute that '{ground_truth}' is accurate. Historical logs indicate synthetic alternative facts.",
        score=0.99
    )
    return [conflict_doc] + [c for c in contexts]


@app.get("/api/health")
def health_check():
    """Returns backend status, active embedding models, and vector retrieval capabilities."""
    return {
        "status": "online",
        "embeddings_active": True,
        "supported_embedding_models": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "ollama/nomic-embed-text",
            "openai/text-embedding-3-small"
        ],
        "supported_llm_models": [
            "ollama/qwen3.5:9b",
            "ollama/llama3",
            "ollama/mistral",
            "gemini-3-flash-preview",
            "synthetic-evaluator"
        ],
        "vector_store": "ChromaDB (LangChain)" if CHROMA_AVAILABLE else "Dense Vector Distance",
        "ollama_support": True,
        "openai_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE")
    }


@app.get("/api/dataset")
def get_dataset():
    """Serves the HotpotQA benchmark evaluation questions."""
    ds = load_hotpotqa_dataset()
    return {"dataset": ds, "total": len(ds)}


@app.post("/api/ingest")
def trigger_ingestion(req: Optional[IngestRequest] = None):
    """Triggers the modular HotpotQA ingestion utility to dynamically seed ChromaDB."""
    try:
        from ingest import ingest_hotpotqa
        num_rows = req.num_rows if req else 50
        emb_model = req.embedding_model if req else "sentence-transformers/all-mpnet-base-v2"
        chunk_size = req.chunk_size if req else 200
        chunk_overlap = req.chunk_overlap if req else 50
        
        res = ingest_hotpotqa(
            num_rows=num_rows,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=emb_model
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


def evaluate_metrics(req, baseline_query, perturbed_query, baseline_contexts,
                     perturbed_contexts, perturbed_retrieved_contexts,
                     baseline_answer, perturbed_answer,
                     base_diag: Optional[Dict[str, Any]] = None,
                     pert_diag: Optional[Dict[str, Any]] = None,
                     gold_titles: Optional[List[str]] = None):
    ground_truth = req.ground_truth
    resolved_gold = gold_titles if gold_titles is not None else (req.gold_titles or [])

    if base_diag is None:
        base_diag = diagnose_failure(
            retrieved_titles=[c.title for c in baseline_contexts],
            gold_titles=resolved_gold,
            generated_answer=baseline_answer,
            ground_truth=ground_truth
        )
    if pert_diag is None:
        pert_diag = diagnose_failure(
            retrieved_titles=[c.title for c in perturbed_contexts],
            gold_titles=resolved_gold,
            generated_answer=perturbed_answer,
            ground_truth=ground_truth
        )

    # 4. Dense Vector Telemetry & Metric Calculations
    # A. Query Embedding Shift: CosineSim(E(Q_baseline), E(Q_perturbed))
    query_shift = compute_dense_cosine_similarity(baseline_query, perturbed_query, req.embedding_provider)

    # B. Retrieval Rank Shift: Top-K Jaccard Overlap
    rank_shift_jaccard = compute_jaccard_overlap(baseline_contexts, perturbed_retrieved_contexts)

    # C. Context Construction Loss: CosineSim(E(C_baseline), E(C_perturbed))
    base_ctx_str = " ".join([f"{c.title} {c.text}" for c in baseline_contexts])
    pert_ctx_str = " ".join([f"{c.title} {c.text}" for c in perturbed_contexts])
    context_loss_sim = compute_dense_cosine_similarity(base_ctx_str, pert_ctx_str, req.embedding_provider)

    # D. Generation vs Ground Truth Cosine Similarity
    gen_gt_baseline_sim = compute_dense_cosine_similarity(baseline_answer, ground_truth, req.embedding_provider)
    gen_gt_perturbed_sim = compute_dense_cosine_similarity(perturbed_answer, ground_truth, req.embedding_provider)

    # High-level retrieval vs generation similarity
    retrieval_sim = context_loss_sim
    generation_sim = compute_dense_cosine_similarity(baseline_answer, perturbed_answer, req.embedding_provider)

    # E. Mathematical Error Amplification Factor
    retrieval_loss = max(0.0, 1.0 - retrieval_sim)
    generation_loss = max(0.0, 1.0 - generation_sim)
    amplification_factor = generation_loss / (retrieval_loss + 0.00001)

    # Status Classification
    if generation_sim > retrieval_sim + 0.08:
        status_category = "Absorbed"
        status_text = "Absorbed (LLM Resilient)"
        status_desc = "LLM generator compensated for input/context perturbation."
    elif generation_sim < retrieval_sim - 0.08:
        status_category = "Amplified"
        status_text = "Amplified (LLM Output Degraded)"
        status_desc = "Minor perturbation caused significant degradation in answer accuracy."
    else:
        status_category = "Proportional"
        status_text = "Proportional"
        status_desc = "Error level carried through uniformly to final output."

    metrics = MetricTelemetry(
        query_embedding_shift=round(query_shift, 4),
        retrieval_rank_shift_jaccard=round(rank_shift_jaccard, 4),
        context_construction_loss=round(context_loss_sim, 4),
        generation_vs_gt_baseline=round(gen_gt_baseline_sim, 4),
        generation_vs_gt_perturbed=round(gen_gt_perturbed_sim, 4),
        retrieval_sim=round(retrieval_sim, 4),
        generation_sim=round(generation_sim, 4),
        amplification_factor=round(amplification_factor, 2),
        status_category=status_category,
        status_text=status_text,
        status_desc=status_desc,
        context_recall_baseline=base_diag["context_recall"],
        context_recall_perturbed=pert_diag["context_recall"],
        answer_f1_baseline=base_diag["answer_f1"],
        answer_f1_perturbed=pert_diag["answer_f1"],
        failure_attribution=pert_diag["verdict"],
        failure_root_cause=pert_diag["root_cause"],
        context_precision_baseline=base_diag.get("context_precision", 1.0),
        context_precision_perturbed=pert_diag.get("context_precision", 1.0),
    )
    return metrics

def _copy_contexts(contexts):
    return [doc.model_copy(deep=True) for doc in contexts]


async def evaluation_events(req: EvaluationRequest):
    """One baseline per request; each variant starts with independent inputs."""
    import uuid as uuid_module

    methods = ordered_methods(req.perturbation_types or [req.perturbation_type])
    groups = [methods] if req.mode == "combined" else [[method] for method in methods]
    seed = req.seed if req.seed is not None else random.SystemRandom().randint(0, 2147483647)
    evaluation_id = uuid_module.uuid4().hex
    settings = {**req.settings.model_dump(), "typo_rate": req.typo_rate}
    baseline_query = req.custom_query.strip() if req.custom_query and req.custom_query.strip() else req.baseline_query
    clean = _copy_contexts([c for c in req.contexts if "Synthetic" not in c.title])

    gold_titles = list(req.gold_titles) if req.gold_titles else []
    if not gold_titles:
        dataset_lookup = load_hotpotqa_dataset()
        for r in dataset_lookup:
            if (req.question_id and str(r.get("id")) == str(req.question_id)) or \
               (r.get("question", "").strip().lower() == req.baseline_query.strip().lower()) or \
               (r.get("question", "").strip().lower() == baseline_query.strip().lower()):
                gold_titles = r.get("gold_titles", [])
                break

    yield {"event": "progress", "message": "Running shared baseline", "total": len(groups)}
    baseline_contexts = await asyncio.to_thread(
        perform_vector_retrieval, _copy_contexts(clean), baseline_query,
        req.search_type, req.top_k, req.embedding_provider,
    )
    baseline_answer, baseline_warning = await generate_llm_answer(
        query=baseline_query, contexts=_copy_contexts(baseline_contexts),
        model_choice=req.llm_model, ground_truth=req.ground_truth,
        gemini_api_key=req.gemini_api_key, is_perturbed=False,
    )
    baseline_source = "synthetic" if baseline_warning else req.llm_model
    base_diag = diagnose_failure(
        retrieved_titles=[c.title for c in baseline_contexts],
        gold_titles=gold_titles,
        generated_answer=baseline_answer,
        ground_truth=req.ground_truth
    )
    yield {"event": "baseline", "baseline_query": baseline_query,
           "baseline_contexts": [c.model_dump() for c in baseline_contexts],
           "baseline_answer": baseline_answer, "baseline_generation_source": baseline_source,
           "gold_titles": gold_titles, "baseline_diagnosis": base_diag}

    dataset = load_hotpotqa_dataset() if {"irrelevant", "remove_evidence"}.intersection(methods) else []
    supporting_titles = req.supporting_titles
    if supporting_titles is None and req.question_id and not req.custom_query:
        row = next((r for r in dataset if str(r.get("id")) == str(req.question_id)), {})
        supporting_titles = row.get("supporting_titles")
        facts = row.get("supporting_facts")
        if supporting_titles is None and isinstance(facts, dict):
            supporting_titles = facts.get("title")
    pool = distractor_pool(dataset, req.question_id, [c.model_dump() for c in clean], baseline_query)
    variants = []
    for index, selected in enumerate(groups, 1):
        yield {"event": "progress", "message": f"Running variant {index} of {len(groups)}",
               "current": index, "total": len(groups), "methods": selected}
        query, candidates, changes = baseline_query, _copy_contexts(clean), []
        rephrase_warning = None
        if "rephrase" in selected:
            before = query
            query, rephrase_warning = await generate_query_rephrase(query, req.llm_model, req.gemini_api_key)
            changes.append(change_record("rephrase", "pre_retrieval", query != before,
                                         "Rephrased question." if query != before else "Question unchanged.",
                                         [{"original": before, "replacement": query, "warning": rephrase_warning}]))
        if "spelling" in selected:
            before = query
            query = inject_spelling_typos(query, req.typo_rate, method_rng(seed, "spelling"))
            changes.append(change_record("spelling", "pre_retrieval", query != before,
                                         "Injected question typos." if query != before else "No characters changed at this severity/seed.",
                                         [{"original": before, "replacement": query}]))
        if "emoji" in selected:
            query, emoji_change = perturb_query_with_emojis(query, req.settings, seed)
            changes.append(emoji_change)
        if "conflict" in selected:
            candidates = inject_conflicting_context(candidates, query, req.ground_truth)
            changes.append(change_record("conflict", "pre_retrieval", True,
                                         "Added contradictory candidate document; retrieval determines whether it reaches generation.",
                                         [candidates[0].model_dump()]))

        if PRE_RETRIEVAL.intersection(selected):
            retrieved = await asyncio.to_thread(
                perform_vector_retrieval, candidates, query, req.search_type, req.top_k, req.embedding_provider,
            )
        else:
            # Post-retrieval experiments use exactly the baseline retrieval output.
            retrieved = _copy_contexts(baseline_contexts)
        final_docs, context_changes = apply_context_methods(
            [c.model_dump() for c in retrieved], selected, req.settings, seed, supporting_titles, pool,
        )
        changes.extend(context_changes)
        final_contexts = [ContextDoc(**doc) for doc in final_docs]
        answer, answer_warning = await generate_llm_answer(
            query=query, contexts=_copy_contexts(final_contexts), model_choice=req.llm_model,
            ground_truth=req.ground_truth, gemini_api_key=req.gemini_api_key, is_perturbed=True,
            perturbation_type="conflict" if "conflict" in selected else selected[0],
        )
        warning = " | ".join(dict.fromkeys(w for w in [baseline_warning, answer_warning, rephrase_warning] if w)) or None
        pert_diag = diagnose_failure(
            retrieved_titles=[c.title for c in final_contexts],
            gold_titles=gold_titles,
            generated_answer=answer,
            ground_truth=req.ground_truth
        )
        metrics = await asyncio.to_thread(
            evaluate_metrics, req, baseline_query, query, baseline_contexts, final_contexts,
            retrieved, baseline_answer, answer, base_diag, pert_diag, gold_titles
        )
        variant = VariantResult(
            variant_id=f"{evaluation_id}:{index}", methods=selected, application_order=selected,
            baseline_query=baseline_query, perturbed_query=query,
            baseline_contexts=_copy_contexts(baseline_contexts), perturbed_contexts=final_contexts,
            baseline_retrieved_contexts=_copy_contexts(baseline_contexts),
            perturbed_retrieved_contexts=_copy_contexts(retrieved),
            baseline_answer=baseline_answer, perturbed_answer=answer, ground_truth=req.ground_truth,
            metrics=metrics, warning=warning, changes=changes,
            baseline_generation_source=baseline_source,
            perturbed_generation_source="synthetic" if answer_warning else req.llm_model,
        )
        log_stage_run(
            req, baseline_query, query, baseline_contexts, retrieved, baseline_answer, answer, metrics, warning,
            experiment={
                "evaluation_id": evaluation_id, "variant_id": variant.variant_id,
                "perturbation_type": "+".join(selected), "selected_methods": methods,
                "methods": selected, "mode": req.mode, "settings": settings, "seed": seed,
                "application_order": selected, "changes": changes,
                "ground_truth": req.ground_truth,
                "gold_titles": gold_titles,
                "baseline_generation_source": baseline_source,
                "perturbed_generation_source": variant.perturbed_generation_source,
                "context_stage": {
                    "baseline_retrieved": [c.model_dump() for c in baseline_contexts],
                    "perturbed_retrieved": [c.model_dump() for c in retrieved],
                    "baseline_final": [c.model_dump() for c in baseline_contexts],
                    "perturbed_final": final_docs,
                },
            },
            gold_titles=gold_titles
        )
        variants.append(variant)
        yield {"event": "variant", "variant": variant.model_dump()}

    response = EvaluationResponse(
        **variants[0].model_dump(), evaluation_id=evaluation_id, mode=req.mode,
        selected_methods=methods, settings=settings, seed=seed, variants=variants,
    )
    yield {"event": "complete", "result": response.model_dump()}


@app.post("/api/run-evaluation", response_model=EvaluationResponse)
async def run_evaluation(req: EvaluationRequest):
    async for event in evaluation_events(req):
        if event["event"] == "complete":
            return event["result"]


@app.post("/api/run-evaluation/stream")
async def stream_evaluation(req: EvaluationRequest):
    """NDJSON progress/results for the dashboard; validation still returns HTTP 422."""
    async def stream():
        try:
            async for event in evaluation_events(req):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            print(f"Evaluation stream failed: {exc}")
            yield json.dumps({"event": "error", "message": "Evaluation failed. Check the backend terminal for details."}) + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
