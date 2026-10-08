# Perturbation x Repair Comparison

Answer F1 against the ground truth (0 to 1). Recovered = share of the lost F1 the fix won back.

| Perturbation | Fix | Questions | Clean | Broken | Fixed | Change | Recovered | Fix triggered |
|---|---|---|---|---|---|---|---|---|
| alter_facts | fix_alter_facts | 6 | 0.464 | 0.303 | 0.273 | -0.029 | -18% | 0% |
| alter_facts | fix_alter_facts_drop | 5 | 0.499 | 0.321 | 0.156 | -0.165 | -93% | 0% |

Left out: 5 run(s) (synthetic answer).
