# Altered-Facts Repair: Detection Findings (RS-62)

**Methods under test (backend/repairs.py):**
- `fix_alter_facts`: cross-checks numbers across passages, corrects a figure the other passages out-vote, flags the rest
- `fix_alter_facts_drop`: drops the passage whose figure is out-voted or contains an impossible date

**Sample:** 50 questions, seed 42. For each passage-set size the real `alter_facts` perturbation
changed one number or date (43 altered cases at 10 passages). Passages are the question's own candidate
passages, gold passages first.

## Results

| Repair | Passages per case | Altered cases | Cross-checkable | Corrected exactly | Altered passage dropped | Flagged only | Missed | Other passages wrongly changed | Clean cases wrongly changed | Clean cases flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_alter_facts | 2 | 33 | 1 | 0 | 0 | 1 | 32 | 0 | 0/50 | 0/50 |
| fix_alter_facts | 3 | 35 | 0 | 0 | 0 | 0 | 35 | 0 | 0/50 | 0/50 |
| fix_alter_facts | 5 | 38 | 0 | 0 | 0 | 0 | 38 | 0 | 0/50 | 0/50 |
| fix_alter_facts | 10 | 43 | 1 | 0 | 0 | 0 | 43 | 0 | 0/50 | 0/50 |
| fix_alter_facts_drop | 2 | 33 | 1 | 0 | 0 | 1 | 32 | 0 | 0/50 | 0/50 |
| fix_alter_facts_drop | 3 | 35 | 0 | 0 | 0 | 0 | 35 | 0 | 0/50 | 0/50 |
| fix_alter_facts_drop | 5 | 38 | 0 | 0 | 0 | 0 | 38 | 0 | 0/50 | 0/50 |
| fix_alter_facts_drop | 10 | 43 | 1 | 0 | 0 | 0 | 43 | 0 | 0/50 | 0/50 |

- **Cross-checkable** counts altered numbers for which another non-sibling passage that refers to this one (its
  text mentions the other's title) states a number with the same signature (same two preceding words, next word
  and size class). Only these can ever be fixed.
- **Corrected exactly** means the repaired passage text equals the original text before alteration.
- **Clean cases wrongly changed** counts cases where the repair corrected or dropped a passage although nothing was altered.

## What the results say

- **It only works where a fact is repeated.** In HotpotQA a fact usually appears in one passage, so most altered
  numbers have nothing to cross-check against (see the Cross-checkable column). When there is no second passage,
  the repair cannot know the original value.
- **A majority is needed to correct.** With a one-against-one disagreement the repair cannot tell which passage is
  right, so it only flags it. Corrections need at least 2 passages agreeing against 1, which means 3 or more
  retrieved passages. The default top-k of 2 almost never allows it.
- **Impossible dates are the one signal that needs no second passage.** A date like 30 February is flagged (or the
  passage dropped), but the original value cannot be recovered.
- **False positives come from sibling passages.** Passages about similar entities (for example consecutive years of
  an event) can contain the same sentence with different numbers. Titles that differ only by digits are skipped, but
  other look-alike passages can still trigger a wrong correction.

## Examples

- None observed in this run.

## Limitations

- The signature matching is lexical. The same fact worded differently in two passages is not matched.
- Numbers are matched only if they are bounded integers (the same kind `alter_facts` changes). Decimals and numbers
  with thousands separators are ignored.
- Dropping a passage removes all of its evidence, not just the wrong figure, so `fix_alter_facts_drop` can hurt answers.
- This is an offline detection test. Whether answers improve is measured separately with the dashboard pipeline.

## Suggested next steps

- Use a semantic or entity-aware matcher so reworded facts can be compared.
- Retrieve more passages (higher top-k) when a figure disagrees, to obtain a majority.
- Verify disputed figures against a trusted source instead of only against other retrieved passages.
