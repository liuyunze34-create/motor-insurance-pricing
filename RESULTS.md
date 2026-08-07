# Results at a glance

## Portfolio

- 678,013 policy records
- Modeling pure premium: €145.58 per exposure-year
- Positive-loss experience is strongly concentrated: the largest 1% of positive-loss policies account for about 28.7% of modeled claim amount after capping

## Model selection

Model family was selected on repeated validation splits before the final test set was used.

| Model | Mean validation Tweedie deviance | Mean validation Gini |
|---|---:|---:|
| Boosted two-part | **73.714** | **0.346** |
| Two-part GLM | 74.941 | 0.317 |
| Direct Tweedie GLM | 75.028 | 0.308 |

Selected model: **Boosted two-part frequency × severity**.

## Final test

- Tweedie deviance: **77.884**
- D²: **5.99%**
- Gini: **0.320**
- Normalized Gini: **0.325**
- Predicted / actual aggregate claim cost: **0.933**
- Highest / lowest observed risk-decile pure premium: **10.9×**

## Interpretation

The nonlinear model improves risk ranking and aggregate pure-premium fit relative to the GLM benchmarks. The severity component remains weak on an individual-policy basis, so the project does not claim strong claim-size prediction. The main value is the combined portfolio risk segmentation and pricing comparison.
