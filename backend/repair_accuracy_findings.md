# Conflicting-Context Repair Method — Detector Accuracy Findings

**Method under test:** `contradiction_filter` (backend/repairs.py)
**Sample size:** 20 questions x 3 cases (clean / conflict / metadata_injection)

## Results

| Metric | Value |
|---|---|
| True Positives | 40 |
| False Negatives | 0 |
| False Positives | 0 |
| True Negatives | 600 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| Accuracy | 1.000 |

## How the detector works

Two signals, checked per document:
1. **Title prefix "Synthetic"** — a strong, cheap signal, but it's just this
   framework's own internal naming convention for injected test documents.
   Relying on it alone would trivially score ~100% here and wouldn't tell us
   anything about real-world detection.
2. **Suspicious phrasing in the document text** — regex match on injected
   wording used by our own perturbations (`"conflicting modern sources"`,
   `"refute that"`, `"alternative facts"`, `"[Topic tags:"`). This is the
   more meaningful signal, since it's the kind of heuristic that could
   plausibly generalize beyond our own naming convention.

## Limitations

- This is a **lexical/pattern-based** detector, not a semantic one. A
  differently-worded conflicting document (e.g. one that contradicts the
  ground truth without using words like "refute" or "conflicting") would
  likely evade it — a real false negative risk in production use, not just
  in this synthetic test.
- The "clean" case only measures false positives against real HotpotQA
  passages, which are unlikely to naturally contain phrases like "conflicting
  modern sources." A more adversarial false-positive test (e.g. real
  passages that legitimately discuss disputes/contradictions/retractions)
  would be a more meaningful next step.
- The metadata-injection signal check (`"[Topic tags:"`) is tightly coupled
  to our own injection format in `perturbations.py`; if that format changes,
  this detector needs to be updated alongside it.

## Suggested next steps

- Add an adversarial false-positive test set (real documents about disputed
  facts, contradicted claims, or retracted findings) to get a more honest
  precision estimate.
- Consider a semantic/NLI-based check as a second layer for documents that
  pass the lexical filter, to catch reworded conflicting content.
