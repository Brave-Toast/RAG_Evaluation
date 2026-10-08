# Repair Method: Altered Facts Fix (RS-62)

## What was built

Two repairs in `backend/repairs.py`, registered so they appear in the dashboard's Repairs section and in
`run_comparison.py`:

- **`fix_alter_facts`** cross-checks numbers and dates across the retrieved passages. If one passage disagrees
  with a majority of at least two other passages, its figure is corrected to the majority value. A one-against-one
  disagreement is only flagged, because nobody can tell which side is right. Impossible dates (30 February) and
  reversed year ranges (1850-1846) are flagged.
- **`fix_alter_facts_drop`** does the same cross-check but drops the passage whose figure is out-voted or contains
  an impossible date.

Two guards limit false alarms. Passages whose titles differ only by digits (for example "2014 ..." and "2015 ...")
are never compared. Only passages that refer to each other (one mentions the other's title) are compared, because
unrelated passages that share a sentence shape ("born April 1971" vs "born April 1975") produced false conflicts.

## Where it works and where it does not

**Offline detection test** (`test_fix_alter_facts.py`, 50 questions, passages per case 2, 3, 5 and 10; the real
`alter_facts` perturbation changed one number per case):

| Passages per case | Altered cases | Could be cross-checked | Corrected | Flagged only | Missed | Clean cases changed or flagged |
|---|---|---|---|---|---|---|
| 2 | 33 | 1 | 0 | 1 | 32 | 0/50 |
| 3 | 35 | 0 | 0 | 0 | 35 | 0/50 |
| 5 | 38 | 0 | 0 | 0 | 38 | 0/50 |
| 10 | 43 | 1 | 0 | 0 | 43 | 0/50 |

Both repairs gave identical results.

- **Safe:** the repair never changed, dropped or flagged a passage when nothing had been altered, and it never touched
  an unaltered passage in an altered case.
- **Almost never applicable:** only 2 of 149 altered numbers had another linked passage stating the same fact. In
  HotpotQA a fact usually appears in one passage, so there is nothing to compare against. This is the limit the ticket
  predicted.
- **It does work when the fact is repeated.** Unit tests with three linked passages stating the same date show the
  odd figure being corrected back to the majority value, or the passage being dropped by the drop variant.
- **Needs three or more passages for a correction.** With the default top-k of 2 a majority is impossible.

## Do answers improve?

Comparison through the dashboard pipeline (`run_comparison.py`), 10 questions requested, top-k 5, embedding model
all-mpnet-base-v2, LLM qwen3.5:9b (local). The same seed was used with and without the fix, so the altered input was
identical. Answer F1 is measured against the ground truth.

| Fix | Questions counted | Clean F1 | After alter_facts, no fix | After alter_facts, with fix | Fix triggered |
|---|---|---|---|---|---|
| fix_alter_facts | 6 | 0.464 | 0.303 | 0.273 | 0% |
| fix_alter_facts_drop | 5 | 0.499 | 0.321 | 0.156 | 0% |

Five runs were left out because the local model timed out and the backend substituted a scripted answer.

- **`alter_facts` hurts answers:** F1 fell by roughly 0.16 to 0.18 from the clean question.
- **The repairs did not change any passage in the counted questions (0% triggered).** The model therefore received
  exactly the same passages with and without the fix. The F1 differences (-0.03 and -0.17) come from the model
  answering the same input differently on two runs, not from the repair, and should not be read as the repair
  making answers worse.
- **No evidence that the repairs improve answers.** With 5 to 6 questions, run-to-run variation in the model's
  answers is larger than any effect we could detect, and the repair almost never acted.

## Limitations

- Signature matching is lexical: the same fact worded differently in two passages is not matched.
- Only bounded integers are matched (the kind `alter_facts` changes). Decimals and numbers with thousands separators are ignored.
- Dropping a passage removes all of its evidence, not just the wrong figure.
- The answers comparison is small (5 to 6 questions after exclusions), uses a single local model, and one run per condition.

## Suggested next steps

- Match facts with an entity-aware or semantic method, so reworded facts can be compared.
- When a figure disagrees, retrieve more passages to get a majority.
- Verify disputed figures against a trusted source instead of only against other retrieved passages.
- Repeat the comparison with a faster or hosted model so more questions finish, and with several runs per question.

## How to reproduce

```
cd backend
python test_fix_alter_facts.py
python run_comparison.py --perturbations alter_facts --pair alter_facts=fix_alter_facts_drop --questions 10 --top-k 5 --output ../results/alter_facts_topk5.jsonl
python build_comparison_table.py --input ../results/alter_facts_topk5.jsonl --output-dir ../results/alter_facts_topk5
```
