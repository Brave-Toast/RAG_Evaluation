# Perturbation x Repair Comparison

Accuracy = answers with F1 >= 0.5 out of all questions. F1 = word overlap with the ground truth (0 to 1). Recovery Rate = share of the lost F1 the repair won back. Failure Stage = where the swap test found the answer broke (no repair).

| Perturbation | Repair Method | Sample Size | Baseline Accuracy | Perturbed Accuracy | Repaired Accuracy | Baseline F1 | Perturbed F1 | Repaired F1 | F1 Gain | Recovery Rate | Repair Trigger Rate | Failure Stage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alter_facts | fix_alter_facts | 52 | 39/52 | 38/52 | 38/52 | 0.726 | 0.711 | 0.711 | +0.0 | - | 0% | CONTEXT 1 |
| alter_facts | fix_alter_facts_drop | 52 | 39/52 | 38/52 | 38/52 | 0.726 | 0.711 | 0.711 | +0.0 | - | 0% | CONTEXT 1 |
| conflict | contradiction_filter | 80 | 60/80 | 17/80 | 47/80 | 0.722 | 0.21 | 0.558 | +0.348 | 68% | 100% | RETRIEVAL 43 |
| conflict | contradiction_filter_text_only | 80 | 60/80 | 17/80 | 17/80 | 0.722 | 0.21 | 0.21 | +0.0 | 0% | 0% | RETRIEVAL 43 |
| emoji | (no repair available) | 80 | 60/80 | 57/80 | - | 0.722 | 0.687 | - | - | - | - | RETRIEVAL 3, QUERY 1 |
| entity_substitution | (no repair available) | 68 | 50/68 | 38/68 | - | 0.707 | 0.545 | - | - | - | - | QUERY 10, RETRIEVAL 5 |
| irrelevant | (no repair available) | 80 | 60/80 | 58/80 | - | 0.722 | 0.697 | - | - | - | - | CONTEXT 2 |
| negation_flip | (no repair available) | 68 | 52/68 | 35/68 | - | 0.738 | 0.498 | - | - | - | - | QUERY 17 |
| remove_evidence | (no repair available) | 80 | 60/80 | 42/80 | - | 0.722 | 0.504 | - | - | - | - | CONTEXT 18 |
| rephrase | (no repair available) | 80 | 60/80 | 57/80 | - | 0.722 | 0.689 | - | - | - | - | RETRIEVAL 4, QUERY 4 |
| shuffle | (no repair available) | 80 | 60/80 | 54/80 | - | 0.722 | 0.662 | - | - | - | - | CONTEXT 6 |
| spelling | (no repair available) | 74 | 55/74 | 46/74 | - | 0.72 | 0.618 | - | - | - | - | RETRIEVAL 7, QUERY 4 |
| truncate | (no repair available) | 80 | 60/80 | 49/80 | - | 0.722 | 0.593 | - | - | - | - | CONTEXT 11 |

Left out: 58 run(s) (perturbation changed nothing).
