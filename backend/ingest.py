"""
Modular Data Ingestion and Vector Storage Seeding Utility for HotpotQA RAG Pipeline.

Extracts HotpotQA distractor dataset from Hugging Face, structures and chunks
context paragraphs using RecursiveCharacterTextSplitter (chunk_size=200, chunk_overlap=50),
computes dense embeddings with sentence-transformers/all-mpnet-base-v2, and seeds ChromaDB.
"""

import os
import sys
import json
import random
import shutil
import pathlib
import argparse
from typing import List, Dict, Tuple, Any, Optional

# Attempt imports for LangChain, HuggingFace datasets, Chroma, and Text Splitters
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
        # Minimal recursive character text splitter fallback
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50, separators: Optional[List[str]] = None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_text(self, text: str) -> List[str]:
                if len(text) <= self.chunk_size:
                    return [text]
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    if end >= len(text):
                        chunks.append(text[start:])
                        break
                    # Find last space or newline before chunk limit
                    split_idx = text.rfind(" ", start, end)
                    if split_idx == -1 or split_idx <= start:
                        split_idx = end
                    chunks.append(text[start:split_idx].strip())
                    start = max(start + 1, split_idx - self.chunk_overlap)
                return [c for c in chunks if c]

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HF_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        CHROMA_AVAILABLE = True
    except ImportError:
        CHROMA_AVAILABLE = False


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_PERSIST_DIR = pathlib.Path(__file__).parent / "Chroma_DB"
DEFAULT_DATASET_JSON = pathlib.Path(__file__).parent / "hotpotqa_dataset.json"


def load_data(split: str = "train"):
    """
    Loads the HotpotQA dataset ('distractor' configuration) from Hugging Face Hub.
    
    The 'distractor' configuration contains questions with 10 candidate paragraphs
    (supporting paragraphs mixed with distractor passages) to evaluate RAG retrieval.
    
    Args:
        split: Dataset split to load ('train' or 'validation').
        
    Returns:
        The loaded Hugging Face dataset dictionary/split.
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "The 'datasets' package is required to load HotpotQA from Hugging Face. "
            "Install it via: pip install datasets"
        )
    
    print(f"Loading 'hotpotqa/hotpot_qa' (distractor split: {split}) from Hugging Face Hub...")
    try:
        data = load_dataset("hotpotqa/hotpot_qa", "distractor")
        print(f"HotpotQA dataset loaded successfully. Available splits: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}", file=sys.stderr)
        raise


def extract_paragraphs(data, num_rows: int = 50, split: str = "train") -> List[str]:
    """
    Extracts and joins context titles and sentence lists into coherent paragraphs.
    
    Args:
        data: The Hugging Face dataset or dictionary.
        num_rows: Number of question rows to extract paragraphs from.
        split: Dataset split to use ('train' or 'validation').
        
    Returns:
        List of formatted paragraph strings: "Title : Sentence1Sentence2..."
    """
    paragraphs = []
    dataset_split = data[split] if hasattr(data, "__getitem__") and split in data else data
    available_rows = min(num_rows, len(dataset_split))
    
    print(f"Extracting paragraphs from {available_rows} rows of HotpotQA ({split})...")
    for i in range(available_rows):
        row = dataset_split[i]
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for title, sent_list in zip(titles, sentences):
            joined_sentence = "".join(sent_list).strip()
            if joined_sentence:
                full_text = f"{title} : {joined_sentence}"
                paragraphs.append(full_text)
                
    print(f"Extracted {len(paragraphs)} raw paragraphs.")
    return paragraphs


def chunk_paragraphs(paragraphs: List[str], chunk_size: int = 200, chunk_overlap: int = 50) -> List[str]:
    """
    Chunks a list of paragraphs using RecursiveCharacterTextSplitter.
    
    Args:
        paragraphs: List of raw paragraph texts.
        chunk_size: Maximum characters per chunk (default: 200).
        chunk_overlap: Overlapping characters between adjacent chunks (default: 50).
        
    Returns:
        List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for paragraph in paragraphs:
        temp_chunks = text_splitter.split_text(paragraph)
        chunks.extend(temp_chunks)
        
    print(f"Chunked {len(paragraphs)} paragraphs into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Initializes and returns the HuggingFaceEmbeddings instance for dense vector encoding.
    
    Args:
        model_name: Hugging Face model identifier (default: sentence-transformers/all-mpnet-base-v2).
        
    Returns:
        Embeddings model instance.
    """
    print(f"Initializing embedding model: '{model_name}'...")
    if HF_EMBEDDINGS_AVAILABLE:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    else:
        # Fallback to langchain_community or direct sentence_transformers wrapper
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHFEmbeddings
            embedding_model = CommunityHFEmbeddings(model_name=model_name)
        except Exception:
            from sentence_transformers import SentenceTransformer
            class STEmbeddingsWrapper:
                def __init__(self, name):
                    self.st = SentenceTransformer(name)
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.st.encode(texts).tolist()
                def embed_query(self, text: str) -> List[float]:
                    return self.st.encode(text).tolist()
            embedding_model = STEmbeddingsWrapper(model_name)
            
    print("Embedding model initialized successfully.")
    return embedding_model


def store_chunks(chunks: List[str], embedding_model, persist_directory: Optional[str]=str(DEFAULT_PERSIST_DIR),clean_exisitng: bool=True):
    if not CHROMA_AVAILABLE:
        raise ImportError( "ChromaDB / LangChain Chroma integration is not installed.\nInstall via: pip install chromadb langchain-chroma")
    #fix is clearing chroma's in-memory client cache so system can not keep reusing a clinet pointed to old dimension
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception as e:
        print(f"Notice: could not clear Chroma System cache: {e}")
    
    if clean_exisitng and persist_directory and os.path.exists(persist_directory):
        print(f"Clearing existing ChromaDB directory at '{persist_directory}'")
        import time
        for attempt in range(3):
            try:
                shutil.rmtree(persist_directory)
                break
            except Exception as e:
                if attempt==2:
                    raise RuntimeError(f"Could not remove '{persist_directory}' after 3 attempts.\nPlease restart the Jupyter kernel and retry): {e}")
                print(f"Retrying attmept {attempt+1}/3: directory issue, waiting......")
                time.sleep(1)

    unique_ids=[str(i) for i in range(len(chunks))]
    print(f"Writing {len(chunks)} chunks to ChromaDB at '{persist_directory}'")

    if persist_directory:
        vector_storage=Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=str(persist_directory), ids=unique_ids)
    else:
        vector_storage=Chroma.from_texts(texts=chunks,embedding=embedding_model,ids=unique_ids)
    #verificvation for the fix now:
    actual_count=vector_storage._collection.count()
    if actual_count!=len(chunks):
        raise RuntimeError(f"Mismatch after ingestion: expected {len(chunks)} chunks.\nCollection has {actual_count}. Old data may not have cleared.")

    print(f"Verified: {actual_count} chunks in collection.")
        
    print("ChromaDB vector store populated successfully.")
    return vector_storage
def extract_and_export_benchmark_json(
    data, 
    num_rows: int = 50, 
    output_json_path: pathlib.Path = DEFAULT_DATASET_JSON,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """
    Extracts structured question-answer pairs and their candidate context passages
    from the HotpotQA dataset and saves them to a clean JSON file for the frontend UI.
    
    Args:
        data: Loaded Hugging Face dataset.
        num_rows: Number of benchmark items to export.
        output_json_path: Path to output JSON file.
        split: Dataset split to use.
        
    Returns:
        List of structured benchmark question items.
    """
    dataset_split = data[split] if hasattr(data, "__getitem__") and split in data else data
    available_rows = min(num_rows, len(dataset_split))
    benchmark_data = []

    for i in range(available_rows):
        row = dataset_split[i]
        q_id = str(row.get("id", i + 1))
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        
        contexts = []
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for title, sent_list in zip(titles, sentences):
            text = "".join(sent_list).strip()
            if text:
                contexts.append({"title": title, "text": text})
                
        benchmark_data.append({
            "id": q_id,
            "question": question,
            "ground_truth": answer,
            "contexts": contexts
        })
        
    if output_json_path:
        output_path = pathlib.Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(benchmark_data)} HotpotQA benchmark questions to '{output_path}'.")
        
    return benchmark_data


def get_random_query(data, num_rows: int = 50, split: str = "train") -> Tuple[str, str, int]:
    """
    Selects a random query and ground-truth answer from the loaded dataset.
    
    Args:
        data: Loaded dataset.
        num_rows: Upper bound for random sampling.
        split: Dataset split.
        
    Returns:
        Tuple of (query, correct_answer, row_index).
    """
    dataset_split = data[split] if hasattr(data, "__getitem__") and split in data else data
    max_idx = min(num_rows, len(dataset_split)) - 1
    random_row = random.randint(0, max_idx)
    query = dataset_split[random_row]["question"]
    correct_answer = dataset_split[random_row]["answer"]
    return query, correct_answer, random_row


def ingest_hotpotqa(
    num_rows: int = 50,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    persist_directory: Optional[str] = str(DEFAULT_PERSIST_DIR),
    export_json: bool = True,
    json_path: pathlib.Path = DEFAULT_DATASET_JSON
) -> Dict[str, Any]:
    """
    Main orchestration function for complete HotpotQA ingestion pipeline.
    
    Can be called directly from FastAPI endpoints or background workers to dynamically seed ChromaDB.
    
    Args:
        num_rows: Number of questions/rows to ingest.
        chunk_size: Chunk character limit.
        chunk_overlap: Chunk character overlap.
        embedding_model_name: Model name for embeddings.
        persist_directory: Target ChromaDB directory.
        export_json: Whether to export structured benchmark JSON.
        json_path: Path for exported JSON.
        
    Returns:
        Dictionary with ingestion statistics and status.
    """
    try:
        data = load_data(split="train")
        paragraphs = extract_paragraphs(data, num_rows=num_rows, split="train")
        chunks = chunk_paragraphs(paragraphs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedding_model = get_embedding_model(model_name=embedding_model_name)
        vector_store = store_chunks(chunks, embedding_model, persist_directory=persist_directory)
        
        exported_count = 0
        if export_json:
            exported_items = extract_and_export_benchmark_json(data, num_rows=num_rows, output_json_path=json_path)
            exported_count = len(exported_items)
            
        return {
            "status": "success",
            "rows_ingested": num_rows,
            "paragraphs_extracted": len(paragraphs),
            "chunks_stored": len(chunks),
            "embedding_model": embedding_model_name,
            "persist_directory": str(persist_directory) if persist_directory else None,
            "benchmark_questions_exported": exported_count
        }
    except Exception as e:
        print(f"Ingestion pipeline failed: {e}", file=sys.stderr)
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HotpotQA dataset and seed ChromaDB for RAG evaluation.")
    parser.add_argument("--rows", type=int, default=50, help="Number of HotpotQA rows to ingest (default: 50)")
    parser.add_argument("--chunk-size", type=int, default=200, help="Text chunk size in characters (default: 200)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Text chunk overlap in characters (default: 50)")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help=f"HuggingFace embedding model (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_PERSIST_DIR), help="ChromaDB persistence directory")
    parser.add_argument("--export-json", action="store_true", default=True, help="Export benchmark JSON for UI")
    
    args = parser.parse_args()
    print("=" * 60)
    print("Starting Modular HotpotQA Ingestion Pipeline")
    print("=" * 60)
    result = ingest_hotpotqa(
        num_rows=args.rows,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model_name=args.embedding_model,
        persist_directory=args.persist_dir,
        export_json=args.export_json
    )
    print("\nIngestion Result:")
    print(json.dumps(result, indent=2))
