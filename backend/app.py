"""
FastAPI Backend Server for True RAG Evaluation, Modular Vector Retrieval (ChromaDB),
Local LLM Integration (Ollama), and Granular Dense Vector Telemetry.
"""

import os
import re
import json
import random
import math
import pathlib
import asyncio
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
import httpx

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
if __package__:
    from .llm_judge import (
        DEFAULT_JUDGE_MODEL,
        SUPPORTED_JUDGES,
        evaluate_with_llm_judge,
    )
    from .perturbations import (Method, PRE_RETRIEVAL, PerturbationSettings, ordered_methods,
                                method_rng, change_record, distractor_pool, apply_context_methods, perturb_query_with_emojis,
                                build_metadata_boosted_candidate)
    from .isolation_metrics import (calculate_token_f1, calculate_exact_match,
                                    calculate_context_recall, calculate_context_precision,
                                    diagnose_failure)
    from .repairs import apply_repair_methods, PRE_RETRIEVAL_REPAIRS, REPAIR_REGISTRY
else:
    try:
        from backend.llm_judge import (
            DEFAULT_JUDGE_MODEL,
            SUPPORTED_JUDGES,
            evaluate_with_llm_judge,
        )
    except ImportError:
        from llm_judge import (
            DEFAULT_JUDGE_MODEL,
            SUPPORTED_JUDGES,
            evaluate_with_llm_judge,
        )
    from perturbations import (Method, PRE_RETRIEVAL, PerturbationSettings, ordered_methods,
                               method_rng, change_record, distractor_pool, apply_context_methods, perturb_query_with_emojis,
                               build_metadata_boosted_candidate)
    from isolation_metrics import (calculate_token_f1, calculate_exact_match,
                                  calculate_context_recall, calculate_context_precision,
                                  diagnose_failure)
    from repairs import apply_repair_methods, PRE_RETRIEVAL_REPAIRS, REPAIR_REGISTRY

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

ENSEMBLE_EMBEDDING_PROVIDER = "ensemble/mpnet+nomic"
ENSEMBLE_COMPONENT_PROVIDERS = (
    "sentence-transformers/all-mpnet-base-v2",
    "ollama/nomic-embed-text",
)

#all the rerankers we support, model name -> label for the dashbord dropdown
#add new ones here n evrything else (validation, dropdown) picks them up automaticly
SUPPORTED_RERANKERS={
    "cross-encoder/ms-marco-MiniLM-L-6-v2":"MiniLM-L6 (small, ~90mb)",
    "BAAI/bge-reranker-base":"BGE base (recommended, ~1.1gb)",
    "BAAI/bge-reranker-v2-m3":"BGE v2 m3 (largest, ~2.2gb)",
}
DEFAULT_RERANKER="BAAI/bge-reranker-base"  #wat gets used when reranker is on n no model is picked

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
    perturbation_types: Optional[List[Method]] = Field(default=None, min_length=1, max_length=12)
    mode: Literal["combined", "separate"] = "combined"
    settings: PerturbationSettings = Field(default_factory=PerturbationSettings)
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)
    supporting_titles: Optional[List[str]] = None
    repair_methods: Optional[List[str]] = None
    typo_rate: float = Field(default=0.25, ge=0, le=1)
    embedding_provider: str = "sentence-transformers/all-mpnet-base-v2"
    search_type: str = "similarity"  # "similarity" or "mmr"
    top_k: int = Field(default=2, ge=1, le=5)
    use_reranker: bool = False  #cross encoder rerank after retrieval, off by deflt so old runs dont change
    rerank_fetch_k: int = Field(default=10, ge=1, le=20)  #how many candiates retriver grabs b4 reranker picks top_k
    reranker_model: str = DEFAULT_RERANKER  #wich reranker to use when its on, must be one from SUPPORTED_RERANKERS
    llm_model: str = "ollama/qwen3.5:9b"  # "ollama/qwen3.5:9b", "ollama/llama3", "ollama/mistral", "gemini-3-flash-preview", "synthetic-evaluator"
    gemini_api_key: Optional[str] = None
    @field_validator("reranker_model")
    @classmethod
    def check_reranker(cls,v):
        #reject unknown model names b4 anything runs, otherwise a typo wud try to download some random model
        if v not in SUPPORTED_RERANKERS:
            raise ValueError(f"unknown reranker '{v}', pick one of {list(SUPPORTED_RERANKERS)}")
        return v
    use_llm_judge: bool = False
    judge_model: str = DEFAULT_JUDGE_MODEL

    @model_validator(mode="after")
    def validate_judge_model(self):
        if self.use_llm_judge and self.judge_model not in SUPPORTED_JUDGES:
            raise ValueError(
                f"Unsupported judge_model '{self.judge_model}'. "
                f"Supported: {list(SUPPORTED_JUDGES)}"
            )
        return self


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
    failure_localisation: Optional[Dict[str, Any]]=None


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
                raise RuntimeError(f"Ollama embeddings failed for '{model_name}': {e}") from e
        else:
            raise RuntimeError(f"Ollama model '{model_name}' is not available; refusing to fall back to a different embedding model.")
    if provider and provider.startswith("ollama/"):
        # Reaching this point means Ollama returned no embedding; never fall through to another model.
        raise RuntimeError(f"Ollama returned no embedding for '{provider}'; refusing to fall back to a different model.")
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
    if provider == ENSEMBLE_EMBEDDING_PROVIDER:
        # Ensemble scores are fused after each model computes cosine similarity
        # in its own vector space; MPNet and Nomic vectors are never combined.
        component_scores = [
            compute_dense_cosine_similarity(text1, text2, component_provider)
            for component_provider in ENSEMBLE_COMPONENT_PROVIDERS
        ]
        return float(np.mean(component_scores))
    vec1 = compute_dense_embedding(text1, provider)
    vec2 = compute_dense_embedding(text2, provider)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return float(np.clip(sim, 0.0, 1.0))


def perform_ensemble_retrieval(
    docs: List[ContextDoc], query: str, search_type: str, top_k: int
) -> List[ContextDoc]:
    """Retrieve by averaging MPNet and Nomic cosine scores without mixing vectors."""
    k_val = min(top_k, len(docs))
    query_vectors = {
        provider: compute_dense_embedding(query, provider)
        for provider in ENSEMBLE_COMPONENT_PROVIDERS
    }

    scored = []
    for doc in docs:
        document_text = f"{doc.title} {doc.text}"
        document_vectors = {
            provider: compute_dense_embedding(document_text, provider)
            for provider in ENSEMBLE_COMPONENT_PROVIDERS
        }
        # Each cosine is calculated within its model's own vector space. Only
        # the scalar scores are fused using the required equal-weight mean.
        component_scores = []
        for provider in ENSEMBLE_COMPONENT_PROVIDERS:
            query_vector = query_vectors[provider]
            document_vector = document_vectors[provider]
            query_norm = np.linalg.norm(query_vector)
            document_norm = np.linalg.norm(document_vector)
            if query_norm == 0 or document_norm == 0:
                component_scores.append(0.0)
            else:
                similarity = float(np.dot(query_vector, document_vector) / (query_norm * document_norm))
                component_scores.append(float(np.clip(similarity, 0.0, 1.0)))
        scored.append({
            "doc": doc,
            "score": float(np.mean(component_scores)),
            "text": doc.text,
        })

    if search_type.lower() == "mmr":
        selected = []
        unselected = list(scored)
        lambd = 0.7
        while len(selected) < k_val and unselected:
            best_idx = 0
            best_mmr = -1e9
            for index, candidate in enumerate(unselected):
                max_redundancy = max(
                    (
                        compute_dense_cosine_similarity(
                            candidate["text"], chosen["text"], ENSEMBLE_EMBEDDING_PROVIDER
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                mmr_score = (lambd * candidate["score"]) - ((1.0 - lambd) * max_redundancy)
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = index
            selected.append(unselected.pop(best_idx))
        scored = selected
    else:
        scored.sort(key=lambda item: item["score"], reverse=True)
        scored = scored[:k_val]

    return [
        ContextDoc(title=item["doc"].title, text=item["doc"].text, score=round(item["score"], 4))
        for item in scored
    ]


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

    # The ensemble must bypass Chroma: MPNet and Nomic have incompatible
    # dimensions/vector spaces, so retrieval fuses their scalar cosine scores.
    if embedding_provider == ENSEMBLE_EMBEDDING_PROVIDER:
        return perform_ensemble_retrieval(docs, query, search_type, top_k)
    
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

# reranker stage - cross encoder reads the questoin n doc TOGETHER and scores how well the doc answers it
# retriever is fast but rough (question n doc embeded seperately), reranker is slow but more carefull
# so we only run it on the few candiates the retriever already picked
_RERANKERS={}  #cache by model name so swapping rerankers dosnt reload evry time

def get_reranker(name=DEFAULT_RERANKER):
    #load once per model n keep it in memory, downloads from huggingface first time if not cached
    if name in _RERANKERS:
        return _RERANKERS[name]
    try:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker '{name}'...")
        #max_length 512 keeps the big models fast, hotpot passages r short anyway so nothing gets cut
        _RERANKERS[name]=CrossEncoder(name,max_length=512)
        print(f"reranker '{name}' loaded")
    except Exception as e:
        print(f"Notice: couldnt load reranker '{name}': {e}")
        return None
    return _RERANKERS[name]

def rerank_docs(query:str,docs:List[ContextDoc],top_k:int,model_name:str=DEFAULT_RERANKER)->List[ContextDoc]:
    #scores evry (question,doc) pair then keeps the best top_k
    if not docs:
        return []
    model=get_reranker(model_name)
    if model is None:
        #dont silently skip it, otherwise a run says reranker on but it actualy wasnt n results get mixed up
        raise RuntimeError(f"reranker '{model_name}' not available, refusing to silently skip the rerank stage")
    #text only, same as wat chroma embeds for the retriver, otherwise reranker sees extra stuff (incl test labels in titles) the retriver never saw
    pairs=[(query,d.text) for d in docs]
    raw_scores=model.predict(pairs)
    ranked=sorted(zip(docs,raw_scores),key=lambda x:float(x[1]),reverse=True)[:min(top_k,len(docs))]
    #some models (ms marco) give raw logits that can be negitive or >1, others (bge) already come out as 0-1 probs
    #only squash the logit ones, otherwise bge scores get sigmoided twice n all show as 0.731
    is_logits=any(float(s)<0 or float(s)>1 for s in raw_scores)
    out=[]
    for d,s in ranked:
        prob=1.0/(1.0+math.exp(-float(s))) if is_logits else float(s)
        out.append(ContextDoc(title=d.title,text=d.text,score=round(prob,4)))
    return out

def retrieve_docs(docs,query,req):
    #single entry point for retrieval so baseline n perturbed runs go thru the exact same stages
    #reranker off = normal retrieval, returns (top_k docs, None)
    #reranker on = grab fetch_k candiates first then reranker picks top_k, returns (top_k docs, the candiates b4 rerank)
    if not req.use_reranker:
        return perform_vector_retrieval(docs,query,req.search_type,req.top_k,req.embedding_provider),None
    fetch_k=max(req.rerank_fetch_k,req.top_k)
    pool=perform_vector_retrieval(docs,query,req.search_type,fetch_k,req.embedding_provider)
    return rerank_docs(query,pool,req.top_k,req.reranker_model),pool

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
    # same prompt for clean n perturbed runs, otherwise were not compareing fairly
    # (old perturbed prompt told the model it was perturbed + to explain conflicts which made answers longer n tanked f1)
    system_prompt = "You are a precise QA assistant. Answer the question using only the provided context. Reply with the short answer only, a few words, no explanation."
    prompt = f"Context:\n{ctx_text}\n\nQuestion: {query}\nShort answer:"
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
                    payload={
                        "model": model_name,
                        "prompt": f"{system_prompt}\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "num_predict": 8000,
                            "temperature":0 #no randomnes so same test gives same anwser evry time
                        }
                    }
                    # only thinking models (qwen3, deepseek-r1) get think off, other models dont support it n might error
                    if "qwen3" in model_name or "deepseek-r1" in model_name:
                        payload["think"]=False
                    res = await client.post(ollama_url, json=payload)
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
                payload={
                    "model": model_name,
                    "prompt": f"{rephrase_system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "num_predict": 8000,
                        "temperature":0 #same rephrase evry run
                    }
                }
                #only thinkin models get think off, same as the anwser gen
                if "qwen3" in model_name or "deepseek-r1" in model_name:
                    payload["think"]=False
                res = await client.post(ollama_url, json=payload)
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

NEGATION_AUX_PATTERN = re.compile(
    r"\b(is|was|are|were|did|does|do|can|could|will|would|has|have|had)\b", re.IGNORECASE
)
NEGATION_WORD_PATTERN = re.compile(r"\bnot\b|n't", re.IGNORECASE)

def inject_negation_flip(text: str) -> tuple[str, bool]:
    """Flips a question's polarity: removes an existing negation, or inserts one after the first auxiliary verb."""
    existing = NEGATION_WORD_PATTERN.search(text)
    if existing:
        flipped = (text[:existing.start()] + text[existing.end():]).replace("  ", " ").strip()
        return flipped, True
    aux = NEGATION_AUX_PATTERN.search(text)
    if aux:
        flipped = text[:aux.end()] + " not" + text[aux.end():]
        return flipped, True
    return text, False
def inject_entity_substitution(text: str, candidate_titles: List[str], replacement_pool: List[Dict[str, Any]], rng) -> tuple[str, Optional[Dict[str, Any]]]:
    """Swaps a named entity (a known document title mentioned in the question) for an unrelated one from another benchmark question."""
    matches = [t for t in candidate_titles if t and t.lower() in text.lower()]
    if not matches or not replacement_pool:
        return text, None
    original_entity = rng.choice(matches)
    replacement_entity = rng.choice(replacement_pool)["title"]
    pattern = re.compile(re.escape(original_entity), re.IGNORECASE)
    new_text = pattern.sub(replacement_entity, text, count=1)
    if new_text == text:
        return text, None
    return new_text, {"original_entity": original_entity, "replacement_entity": replacement_entity}
def inject_conflicting_context(contexts: List[ContextDoc], query: str, ground_truth: str, wrong_answer: Optional[str]=None) -> List[ContextDoc]:
    """Injects a high-priority synthetic passage that claims a wrong answer."""
    #old version put the real anwser in the fake doc so the modle just read it out of it, now it claims a wrong one
    fake=wrong_answer or "an unrelated person"
    conflict_doc = ContextDoc(
        title="Synthetic_Conflicting_Record",
        text=f"NOTICE: Regarding the query '{query}', updated records confirm the answer is {fake}. Earlier sources that claim otherwise have been corrected.",
        score=0.99
    )
    return [conflict_doc] + [c for c in contexts]


def _ollama_reachable() -> bool:
    """True if the local Ollama service answers (used by /api/health)."""
    try:
        with httpx.Client(timeout=1.0) as client:
            return client.get("http://localhost:11434/api/tags").status_code == 200
    except Exception:
        return False


@app.get("/api/health")
def health_check():
    """Returns backend status, active embedding models, and vector retrieval capabilities."""
    return {
        "status": "online",
        "embeddings_active": True,
        "supported_embedding_models": [
            ENSEMBLE_EMBEDDING_PROVIDER,
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
        "ollama_support": _ollama_reachable(),
        "openai_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE"),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))
    }

@app.get("/api/repairs")
def list_repairs():
    """Lists registered repair methods so the dashboard shows one checkbox per fix automatically."""
    repairs = []
    for name, fn in REPAIR_REGISTRY.items():
        if name == "noop":  # test-only repair, keep it off the dashboard
            continue
        doc_lines = (getattr(fn, "__doc__", None) or "").strip().splitlines()
        labels = {
            "contradiction_filter": "Contradiction Filter",
            "fix_shuffle": "Fix Shuffled Documents",
            "fix_truncate": "Fix Truncated Documents",
        }
        repairs.append({
            "name": name,
            "label": labels.get(name, name.replace("_", " ").title()),
            "description": doc_lines[0] if doc_lines else "",
        })
    return {"repairs": sorted(repairs, key=lambda r: r["name"])}

@app.get("/api/rerankers")
def list_rerankers():
    #dashbord calls this to fill the reranker dropdown, same idea as /api/repairs
    return {
        "default": DEFAULT_RERANKER,
        "rerankers": [{"name": name, "label": label} for name, label in SUPPORTED_RERANKERS.items()],
    }
@app.get("/api/judges")
async def list_judges():
    return {
        "default": DEFAULT_JUDGE_MODEL,
        "judges": [
            {"name": "ollama/qwen3.5:9b", "label": "Ollama / qwen3.5:9b (Local Judge)"},
            {"name": "ollama/llama3", "label": "Ollama / llama3 (Local Judge)"},
            {"name": "ollama/mistral", "label": "Ollama / mistral (Local Judge)"},
            {"name": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Cloud Judge)"},
            {"name": "synthetic-evaluator", "label": "Synthetic Heuristic Judge (Fast Offline)"},
        ],
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
    # If the final context is effectively unchanged, the ratio is meaningless (answer drift comes from
    # model/prompt noise, not the perturbation), so report 0 instead of an exploding value.
    amplification_factor = 0.0 if retrieval_loss < 0.01 else min(generation_loss / retrieval_loss, 50.0)

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

def is_correct(answer,ground_truth):
    #quick chek for now, f1>=0.5 coutns as rigth (same cutof as diagnose_failure) can swap fr llm judge latr
    return calculate_token_f1(answer,ground_truth)>=0.5


async def localise_failure(req,perturbed_query,baseline_contexts,retrieved,baseline_answer,perturbed_answer,context_edited,gold_titles,baseline_pool=None,pert_pool=None):
    #swap tset - put the clen stuff bak one stgae at a tiem n see wich swap fixs the anwser
    #stages it can blame: QUERY, RETRIEVAL, RERANKING (only wen reranker is on), CONTEXT, GENERATION
    gt=req.ground_truth

    def has_answer(docs):
        #chek if these docs contain the gold docs (the ones wit the anwser)
        if gold_titles:
            got=[c.title for c in docs]
            return all(t in got for t in gold_titles)
        #custom qustions dont hav gold docs so fal bak to chekin if anwser txt is in teh docs
        return gt.lower() in " ".join(c.text for c in docs).lower()

    async def retrieval_or_reranking(reason,swap_answer):
        #docs were the problem, figure out if the retriver or the reranker picked the wrong ones
        #if reranker was on, try wat the retriver alone wud of sent (top_k of the pool b4 rerank)
        if pert_pool:
            swap3,warn3=await generate_llm_answer(query=perturbed_query,contexts=_copy_contexts(pert_pool[:req.top_k]),
                        model_choice=req.llm_model,ground_truth=gt,gemini_api_key=req.gemini_api_key,is_perturbed=True)
            if warn3:
                return {"failed_at":"UNKNOWN","reason":"model didnt answer during swap test"}
            if is_correct(swap3,gt):
                return {"failed_at":"RERANKING","reason":"the retrievers own picks wud of worked, the reranker swapped in the wrong docs","swap_answer":swap3}
        return {"failed_at":"RETRIEVAL","reason":reason,"swap_answer":swap_answer}

    if not is_correct(baseline_answer,gt):
        #clen run faild to
        if has_answer(baseline_contexts):
            return {"failed_at":"GENERATION","reason":"model had the right docs and clean question but still got it wrong"}
        #reranker on n the retrivers own top_k had the answer docs but the reranker didnt keep them
        if baseline_pool and has_answer(baseline_pool[:req.top_k]):
            return {"failed_at":"RERANKING","reason":"clean run, the retrievers own picks had the answer docs but the reranker dropped them"}
        return {"failed_at":"RETRIEVAL","reason":"even the clean run didnt retrieve the answer docs, retriever missed it"}
    if is_correct(perturbed_answer,gt):
        return {"failed_at":"NONE","reason":"perturbed answer is still correct"}

    #swap 1 keep teh brokn qustion but giv the modle the clen docs
    swap1,warn1=await generate_llm_answer(query=perturbed_query,contexts=_copy_contexts(baseline_contexts),
                    model_choice=req.llm_model,ground_truth=gt,gemini_api_key=req.gemini_api_key,is_perturbed=True)
    if warn1:
        return {"failed_at":"UNKNOWN","reason":"model didnt answer during swap test"}
    if not is_correct(swap1,gt):
        return {"failed_at":"QUERY","reason":"even with clean docs the broken question still gets it wrong","swap_answer":swap1}

    #clen docs fixd it so docs wre the problme, now chek if its retreival/reranking or teh contxt edits
    if not context_edited:
        return await retrieval_or_reranking("clean docs fixed it, retriever picked the wrong docs",swap1)

    #swap 2 giv it the docs as retreived, befor shufle/trunacte etc mesed wit them
    swap2,warn2=await generate_llm_answer(query=perturbed_query,contexts=_copy_contexts(retrieved),
          model_choice=req.llm_model,ground_truth=gt,gemini_api_key=req.gemini_api_key,is_perturbed=True)
    if warn2:
        return {"failed_at":"UNKNOWN","reason":"model didnt answer during swap test"}
    if is_correct(swap2,gt):
        return {"failed_at":"CONTEXT","reason":"retrieved docs were fine, the context edits broke it","swap_answer":swap2}
    return await retrieval_or_reranking("retriever picked the wrong docs",swap2)
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
    #goes thru retrieve_docs so baseline gets reranked too when its on, otherwise comparision isnt fair
    baseline_contexts,baseline_pool = await asyncio.to_thread(
        retrieve_docs, _copy_contexts(clean), baseline_query, req,
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

    dataset = load_hotpotqa_dataset() if {"irrelevant", "remove_evidence", "entity_substitution", "metadata_injection", "conflict"}.intersection(methods) else []
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
        if "negation_flip" in selected:
            before = query
            query, flipped = inject_negation_flip(query)
            changes.append(change_record("negation_flip", "pre_retrieval", flipped,
                                         "Flipped question polarity." if flipped else "No negation word or auxiliary verb found; question unchanged.",
                                         [{"original": before, "replacement": query}]))
        if "entity_substitution" in selected:
            before = query
            candidate_titles = [c.title for c in clean] + gold_titles
            query, sub_detail = inject_entity_substitution(query, candidate_titles, pool, method_rng(seed, "entity_substitution"))
            changes.append(change_record("entity_substitution", "pre_retrieval", sub_detail is not None,
                                         "Substituted a named entity in the question." if sub_detail else "No known entity found in the question, or no distractor pool available; question unchanged.",
                                         [sub_detail] if sub_detail else []))
        pre_retrieval_repairs = [
            method for method in (req.repair_methods or []) if method in PRE_RETRIEVAL_REPAIRS
        ]
        post_retrieval_repairs = [
            method for method in (req.repair_methods or []) if method not in PRE_RETRIEVAL_REPAIRS
        ]
        if pre_retrieval_repairs:
            query, _, repair_changes = apply_repair_methods(
                query, [], pre_retrieval_repairs,
                {"protected_entities": [c.title for c in clean] + gold_titles},
            )
            changes.extend(repair_changes)
        if "conflict" in selected:
            gt = req.ground_truth.strip().lower()
            if gt in ("yes", "no"):
                wrong = "no" if gt == "yes" else "yes"
            else:
                others = [
                    r.get("ground_truth")
                    for r in dataset
                    if r.get("ground_truth")
                    and r.get("ground_truth").strip().lower() != gt
                ]
                wrong = method_rng(seed, "conflict").choice(others) if others else None
            candidates = inject_conflicting_context(
                candidates, query, req.ground_truth, wrong
            )
            changes.append(change_record(
                "conflict", "pre_retrieval", True,
                "Added contradictory candidate document; retrieval determines whether it reaches generation.",
                [candidates[0].model_dump()]
            ))
        if "metadata_injection" in selected:
            meta_candidate, meta_change = build_metadata_boosted_candidate(
                query, pool, [c.model_dump() for c in candidates], req.settings, method_rng(seed, "metadata_injection"),
            )
            changes.append(meta_change)
            if meta_candidate:
                candidates = [ContextDoc(**meta_candidate)] + candidates
        if PRE_RETRIEVAL.intersection(selected) or pre_retrieval_repairs:
                # Re-run retrieval for pre-retrieval perturbations and repairs.
                retrieved, pert_pool = await asyncio.to_thread(
                retrieve_docs, candidates, query, req,
            )
        else:
            # Post-retrieval experiments reuse the baseline retrieval results.
            retrieved = _copy_contexts(baseline_contexts)
            pert_pool = baseline_pool
         final_docs, context_changes = apply_context_methods(
            [c.model_dump() for c in retrieved], selected, req.settings, seed, supporting_titles, pool,
        )
        changes.extend(context_changes)
        final_contexts = [ContextDoc(**doc) for doc in final_docs]
        if post_retrieval_repairs:
            repair_config = {
                # apply_context_methods() mutates copies, so this is the exact
                # legitimate post-retrieval full-text source for restoration.
                "full_contexts": [c.model_dump() for c in retrieved],
            }
            query, repaired_docs, repair_changes = apply_repair_methods(
                query, [c.model_dump() for c in final_contexts], post_retrieval_repairs, repair_config
            )
            final_contexts = [ContextDoc(**doc) for doc in repaired_docs]
            changes.extend(repair_changes)
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
        #chek if the contxt edits actualy chnaged anyting (shufle/truncate etc)
        context_edited=[(d["title"],d["text"]) for d in final_docs]!=[(c.title,c.text) for c in retrieved]
        #swap tset to fnd wich stage broke teh anwser, skip if modle didnt answr
        if answer_warning or baseline_warning:
            localisation={"failed_at":"UNKNOWN","reason":"model didnt answer so cant localise"}
        else:
            localisation=await localise_failure(req,query,baseline_contexts,retrieved,baseline_answer,answer,context_edited,gold_titles,baseline_pool,pert_pool)
        print(f"failure localisation: {localisation}")
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
            failure_localisation=localisation,
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
                #rerank stage - wat the retriver grabbed b4 rerank (pool) vs wat the reranker actualy kept
                #pool scores r retriver scores, kept docs r the ones that went to the llm
                "rerank_stage": {
                    "enabled": req.use_reranker,
                    "model": req.reranker_model if req.use_reranker else None,
                    "fetch_k": req.rerank_fetch_k if req.use_reranker else None,
                    "baseline_pool": [{"title":c.title,"score":c.score} for c in baseline_pool] if baseline_pool else None,
                    "baseline_kept": [c.title for c in baseline_contexts],
                    "perturbed_pool": [{"title":c.title,"score":c.score} for c in pert_pool] if pert_pool else None,
                    "perturbed_kept": [c.title for c in retrieved],
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

try:
    from backend.llm_judge import register_llm_judge
except ImportError:
    try:
        from .llm_judge import register_llm_judge
    except ImportError:
        from llm_judge import register_llm_judge

register_llm_judge(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
