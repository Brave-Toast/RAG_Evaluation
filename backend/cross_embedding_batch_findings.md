# Cross-Embedding Batch: mpnet vs nomic-embed-text

10 questions, seed 42, top-k 2, LLM `synthetic-evaluator`.
Every method runs separately from the clean input. All runs are also logged in `stage_run_logs.jsonl`.

| Embedding model | Method | Cases | Query shift | Retrieved overlap | Gold recall (clean -> perturbed) | Injected doc retrieved |
|---|---|---|---|---|---|---|
| sentence-transformers/all-mpnet-base-v2 | spelling | 10 | 0.85 | 0.73 | 0.80 -> 0.60 | - |
| sentence-transformers/all-mpnet-base-v2 | emoji | 10 | 0.89 | 0.80 | 0.80 -> 0.70 | - |
| sentence-transformers/all-mpnet-base-v2 | conflict | 10 | 1.00 | 0.33 | 0.80 -> 0.45 | 10/10 |
| sentence-transformers/all-mpnet-base-v2 | metadata_injection | 10 | 1.00 | 0.93 | 0.80 -> 0.75 | 1/10 |
| ollama/nomic-embed-text | spelling | 10 | 0.87 | 0.70 | 0.70 -> 0.65 | - |
| ollama/nomic-embed-text | emoji | 10 | 0.93 | 0.87 | 0.70 -> 0.65 | - |
| ollama/nomic-embed-text | conflict | 10 | 1.00 | 0.33 | 0.70 -> 0.45 | 10/10 |
| ollama/nomic-embed-text | metadata_injection | 10 | 1.00 | 0.87 | 0.70 -> 0.70 | 2/10 |

- **Query shift:** cosine similarity between the original and perturbed question embeddings (1.00 = unchanged).
- **Retrieved overlap:** Jaccard overlap between the clean and perturbed retrieved document sets.
- **Injected doc retrieved:** how often the synthetic document reached the top-k (conflict and metadata injection only).
- Answer-based metrics are not meaningful when the LLM is the scripted `synthetic-evaluator`.

## Errors

- None.
