# Metadata Injection and Cross-Embedding Findings (RS-44 / RS-46)

## What was built

- **Metadata injection perturbation** (`perturbations.py`): adds one off-topic passage, tagged with the query's
  salient terms, to the candidate set before retrieval. It is pre-retrieval, so it targets the retrieval stage.
- **Retrieval test** (`test_metadata_injection.py`): for 10 questions it retrieves top-2 from the clean contexts, adds the
  injected chunk, retrieves again, and records whether the injected chunk was retrieved, its rank, and whether a gold
  passage was pushed out. Embedding providers are verified before use, so a "nomic" run cannot silently be another model.
- **Contradiction filter** (`repairs.py`) and its accuracy test: see `repair_accuracy_findings.md`.

## Retrieval test results (10 questions, top-k 2)

| Version | Embedding model | Injected chunk in top-k | Injected at rank 1 | Gold passage pushed out | Gold recall (clean -> injected) |
|---|---|---|---|---|---|
| Original injection | all-mpnet-base-v2 | 0/10 | 0/10 | 0/10 | 0.80 -> 0.80 |
| Original injection | nomic-embed-text | 0/10 | 0/10 | 0/10 | 0.70 -> 0.70 |
| Better terms + repeated tag | all-mpnet-base-v2 | 1/10 | 0/10 | 0/10 | 0.80 -> 0.80 |
| Better terms + repeated tag | nomic-embed-text | 1/10 | 0/10 | 0/10 | 0.70 -> 0.70 |
| + short body, harder distractor | all-mpnet-base-v2 | 1/10 | 0/10 | 1/10 | 0.80 -> 0.75 |
| + short body, harder distractor | nomic-embed-text | 2/10 | 0/10 | 0/10 | 0.70 -> 0.70 |

## What the results say

- **The injected chunk rarely wins retrieval.** With keyword-tag metadata the injected passage reached the top 2
  in at most 2 of 10 cases and never ranked first. The gold passages discuss the same entities as the question,
  and both embedding models weigh that more heavily than tag keywords.
- **Three attempts changed the outcome only slightly.** Ranking the terms, repeating the tag, shortening the
  distractor body and choosing a harder distractor moved the rate from 0/10 to 1-2/10.
- **The effect is similar on both models.** The gap between mpnet (1/10) and nomic (2/10) is one case, which is
  too small to call a real difference. The weak effect does not look specific to one embedding model.
- **When it works it can displace evidence.** On mpnet, one case pushed a gold passage out of the top 2
  (recall 0.80 -> 0.75).
- **Baseline retrieval differs by model.** On these 10 questions, clean gold recall at top-2 was 0.80 for mpnet
  and 0.70 for nomic.

## Cross-embedding batch (live pipeline, 10 questions, top-k 2)

The same perturbations were run through the dashboard backend with each embedding model, one method at a time
from the clean input (`run_nomic_batch.py`; scripted LLM, so only the retrieval columns are meaningful).
Every run is also in `stage_run_logs.jsonl`.

| Embedding model | Method | Query shift | Retrieved overlap | Gold recall (clean -> perturbed) | Injected doc retrieved |
|---|---|---|---|---|---|
| all-mpnet-base-v2 | spelling | 0.85 | 0.73 | 0.80 -> 0.60 | - |
| all-mpnet-base-v2 | emoji | 0.89 | 0.80 | 0.80 -> 0.70 | - |
| all-mpnet-base-v2 | conflict | 1.00 | 0.33 | 0.80 -> 0.45 | 10/10 |
| all-mpnet-base-v2 | metadata_injection | 1.00 | 0.93 | 0.80 -> 0.75 | 1/10 |
| nomic-embed-text | spelling | 0.87 | 0.70 | 0.70 -> 0.65 | - |
| nomic-embed-text | emoji | 0.93 | 0.87 | 0.70 -> 0.65 | - |
| nomic-embed-text | conflict | 1.00 | 0.33 | 0.70 -> 0.45 | 10/10 |
| nomic-embed-text | metadata_injection | 1.00 | 0.87 | 0.70 -> 0.70 | 2/10 |

- **Conflicting context is the strongest perturbation.** The injected document reached the top 2 in 10 of 10 cases
  on both models and cut gold recall from 0.80 to 0.45 (mpnet) and from 0.70 to 0.45 (nomic).
- **Spelling and emoji have a moderate effect.** Gold recall fell by 0.05 to 0.20 on both models.
- **Metadata injection is the weakest.** It was retrieved in 1/10 (mpnet) and 2/10 (nomic) cases, matching the
  offline retrieval test above.
- **The behaviour does not look specific to one embedding model.** Both models rank the methods in the same
  order of severity (conflict, then spelling and emoji, then metadata injection). The remaining differences are
  small and come from only 10 questions, so they should not be read as real differences between the models.
- **Baseline retrieval differs.** On these questions nomic starts at gold recall 0.70 and mpnet at 0.80.

## Contradiction filter in the dashboard

With "Conflicting context" and the "Contradiction filter (repair)" option both enabled, the filter removed
`Synthetic_Conflicting_Record` from the context sent to generation (2 passages became 1). The answer was still
poor ("the provided context does not mention First for Women"), because the injected document had already taken a
top-k slot during retrieval and pushed the real "First for Women" passage out. A post-retrieval filter cannot
restore evidence it displaced, and gold recall stayed at 0.50. Filtering candidates before retrieval would avoid
this and is a suggested follow-up.

## Limits of this test

- 10 questions is a small sample, so the rates are indicative only.
- Each question has about 10 candidate passages and top-k is 2, which is a narrow window. A wider top-k or a
  larger candidate pool may change the rate.
- The injected passage is prefixed with the query's own keywords. This is a deliberately simple attack; a stronger
  variant (for example copying the full question into the header) would probably retrieve better but is closer to
  query copying than metadata.
- The contradiction filter catches these injected documents because they carry the `[Topic tags:` marker. That
  detection is circular, as noted in `repair_accuracy_findings.md`.

## Other fixes made during this work

- Ollama embedding failures now raise an error instead of silently falling back to MiniLM, so a nomic run is a real nomic run.
- Amplification factor is reported as 0 when the final context is unchanged (it previously exploded to values like 13737x).
- Selecting 10 or more perturbation methods no longer returns HTTP 422 (`max_length` 9 -> 12).
- The contradiction filter can be switched on from the dashboard.
- The Root Attribution, Recall and F1 cards show real values instead of fixed defaults.
- Metadata-boosted documents get their own "Metadata Injected" badge, and the nomic option shows "(Ollama not running)" when Ollama is down.

## How to reproduce

```
cd backend
python test_metadata_injection.py --tag <label>
python test_repair_accuracy.py --num-questions 20
python run_nomic_batch.py          # needs the backend and Ollama running
```
