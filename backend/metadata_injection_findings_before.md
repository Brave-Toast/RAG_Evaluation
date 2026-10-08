# Metadata Injection: Retrieval Test Findings

**Perturbation:** `metadata_injection` (backend/perturbations.py), which adds an off-topic passage
carrying query keywords before retrieval.
**Setup:** 10 questions, seed 42, top-k 2, search type `similarity`.
Each case retrieves from the clean contexts, then again with the injected chunk added.

## Results

| Embedding model | Cases | Injected chunk in top-k | Injected at rank 1 | Gold passage pushed out | Gold recall (clean -> injected) |
|---|---|---|---|---|---|
| sentence-transformers/all-mpnet-base-v2 | 10 | 0/10 | 0/10 | 0/10 | 0.80 -> 0.80 |
| ollama/nomic-embed-text | 10 | 0/10 | 0/10 | 0/10 | 0.70 -> 0.70 |

- **Injected chunk in top-k** is the pass criterion for "retrieves the wrong chunk".
- **Gold passage pushed out** counts cases where a gold passage retrieved on the clean run
  disappeared from the top-k after injection.

## Per-case results

| Embedding model | Question | Injected rank | Top-k after injection |
|---|---|---|---|
| sentence-transformers/all-mpnet-base-v2 | What U.S Highway gives access to Zilpo Road, and is also ... | not retrieved | Zilpo Road | Hawaii Route 64 |
| sentence-transformers/all-mpnet-base-v2 | What nationality was James Henry Miller's wife? | not retrieved | June Miller | Ewan MacColl |
| sentence-transformers/all-mpnet-base-v2 | Which documentary is about Finnish rock groups, Adam Clay... | not retrieved | Adam Clayton Powell (film) | The Saimaa Gesture |
| sentence-transformers/all-mpnet-base-v2 | Who was born first, Pablo Trapero or Aleksander Ford? | not retrieved | Pablo Trapero | Aleksander Ford |
| sentence-transformers/all-mpnet-base-v2 | Are both The New Pornographers and Kings of Leon American... | not retrieved | The New Pornographers | Kings of Leon |
| sentence-transformers/all-mpnet-base-v2 | Human Error" is the season finale of the third season of ... | not retrieved | Human Error (House) | Through the Looking Glass (Lost) |
| sentence-transformers/all-mpnet-base-v2 | In which American football game was Malcolm Smith named M... | not retrieved | Malcolm Smith (American football) | Steve Clark (American football) |
| sentence-transformers/all-mpnet-base-v2 | Which city is home to the headquarters of the space agenc... | not retrieved | Chandrayaan-3 | ISRO |
| sentence-transformers/all-mpnet-base-v2 | Gunmen from Laredo starred which narrator of "Frontier"? | not retrieved | Gunmen from Laredo | Walter Coy |
| sentence-transformers/all-mpnet-base-v2 | The person for whom the Hubble Space Telescope was named ... | not retrieved | Edwin Hubble | Hubble Space Telescope |
| ollama/nomic-embed-text | What U.S Highway gives access to Zilpo Road, and is also ... | not retrieved | Zilpo Road | California State Route 266 |
| ollama/nomic-embed-text | What nationality was James Henry Miller's wife? | not retrieved | June Miller | Jim Miller (Australian footballer, born 1919) |
| ollama/nomic-embed-text | Which documentary is about Finnish rock groups, Adam Clay... | not retrieved | The Saimaa Gesture | Adam Clayton Powell (film) |
| ollama/nomic-embed-text | Who was born first, Pablo Trapero or Aleksander Ford? | not retrieved | Pablo Trapero | Aleksander Ford |
| ollama/nomic-embed-text | Are both The New Pornographers and Kings of Leon American... | not retrieved | Kings of Leon | The New Pornographers |
| ollama/nomic-embed-text | Human Error" is the season finale of the third season of ... | not retrieved | Human Error (House) | Through the Looking Glass (Lost) |
| ollama/nomic-embed-text | In which American football game was Malcolm Smith named M... | not retrieved | Malcolm Smith (American football) | Super Bowl Most Valuable Player Award |
| ollama/nomic-embed-text | Which city is home to the headquarters of the space agenc... | not retrieved | Chandrayaan-3 | ISRO |
| ollama/nomic-embed-text | Gunmen from Laredo starred which narrator of "Frontier"? | not retrieved | Gunmen from Laredo | Walter Coy |
| ollama/nomic-embed-text | The person for whom the Hubble Space Telescope was named ... | not retrieved | Edwin Hubble | Marshfield, Missouri |

## Providers not run

- None.

## Notes and limits

- Retrieval embeds the passage text only (the title is metadata on the Chroma path), so the
  injected keywords sit in the text.
- Each question has about 10 candidate passages, so top-k is a small window. Results on a
  larger candidate pool may differ.
- 10 cases is a small sample. Treat the rates as indicative, not precise.
