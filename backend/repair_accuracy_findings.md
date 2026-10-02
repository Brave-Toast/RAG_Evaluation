# Conflicting-Context Repair Method — Detector Accuracy Findings

**Methods under test (backend/repairs.py):**
- `contradiction_filter` ("full": title prefix + text markers)
- `contradiction_filter_text_only` ("text_only": text markers only, no title label)

**Sample:** 20 questions, seed 42. The original test (clean / conflict / metadata_injection)
covers 640 documents, of which 40 are injected. The paraphrased-conflict case and a
3-document adversarial false-positive set were added on top.

## Results

| Filter | Case | TP | FN | FP | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|
| full | clean | 0 | 0 | 0 | 200 | n/a | n/a |
| full | conflict | 20 | 0 | 0 | 200 | 1.000 | 1.000 |
| full | paraphrased_conflict | 0 | 20 | 0 | 200 | n/a | 0.000 |
| full | metadata_injection | 20 | 0 | 0 | 200 | 1.000 | 1.000 |
| full | adversarial_clean | 0 | 0 | 3 | 0 | 0.000 | n/a |
| full | original test (clean+conflict+metadata) | 40 | 0 | 0 | 600 | 1.000 | 1.000 |
| text_only | clean | 0 | 0 | 0 | 200 | n/a | n/a |
| text_only | conflict | 20 | 0 | 0 | 200 | 1.000 | 1.000 |
| text_only | paraphrased_conflict | 0 | 20 | 0 | 200 | n/a | 0.000 |
| text_only | metadata_injection | 20 | 0 | 0 | 200 | 1.000 | 1.000 |
| text_only | adversarial_clean | 0 | 0 | 2 | 1 | 0.000 | n/a |
| text_only | original test (clean+conflict+metadata) | 40 | 0 | 0 | 600 | 1.000 | 1.000 |

## What the results say

- **The original test is easy.** The detector looks for exact strings that our own
  generators insert (the `Synthetic` title, `conflicting modern sources`, `refute that`,
  `alternative facts`, `[Topic tags:`), so a near-perfect score on that test mainly shows
  the filter matches our own generators. With 0 misses out of 40 injected documents, the true recall is only known to be at least about 92% (95% confidence), not exactly 100%.
- **Paraphrased conflicts are the realistic test.** Recall on conflicting documents that avoid those
  keywords was 0.000 (full filter) and 0.000 (text-only filter).
- **False positives on genuine text:** on 3 genuine passages the full filter wrongly removed
  3 and the text-only filter wrongly removed 2. The title rule
  (`startswith("synthetic")`) can remove real articles such as "Synthetic biology".
- **Metadata injection is a different perturbation.** It adds an off-topic passage made to look relevant,
  not a contradiction, so it is reported on its own row.

## Failure examples

- **Missed (full filter, paraphrased_conflict):** title `Regional_Archive_Note`, text: "Recent scholarship on 'What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?' suggests the commonly cited answer, 'US 60', is mistaken. Newly reviewed records point to a different conclusion."
- **Missed (text_only filter, paraphrased_conflict):** title `Regional_Archive_Note`, text: "Recent scholarship on 'What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?' suggests the commonly cited answer, 'US 60', is mistaken. Newly reviewed records point to a different conclusion."
- **Wrongly removed (full filter, adversarial_clean):** a genuine document titled `Synthetic biology`
- **Wrongly removed (text_only filter, adversarial_clean):** a genuine document titled `Cold fusion controversy`

## How the detector works

1. **Title prefix "Synthetic"** — our own internal naming convention for test documents (a label, not content).
2. **Suspicious phrasing in the text** — regex for wording our perturbations insert.

## Limitations

- This is a lexical/pattern-based detector, not a semantic one. It did not detect
  reworded conflicting documents (see the paraphrased_conflict rows).
- The marker list is tied to our own injection formats in `perturbations.py` and `app.py`;
  if those change, the detector must change too.
- The adversarial false-positive set is small and hand-written, so it shows failure modes
  rather than a reliable false-positive rate.

## Suggested next steps

- Add a semantic/NLI-based second layer to catch reworded conflicting content.
- Build a larger false-positive set from real passages about disputes, corrections and retractions.
- Replace the title rule with a content-based check, since the prefix can match real articles.
