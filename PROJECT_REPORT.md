# Motor Insurance Pricing Report

## Data

The analysis uses 678,013 French MTPL policies and a separate claim-severity table. The two sources are joined by policy ID. A reconciliation step is run before modeling because the severity source does not cover every positive claim count in the frequency table.

The raw joined portfolio has 36,102 claims and observed pure premium €167.11. The modeling portfolio has 26,406 claims and observed pure premium €145.58 after the documented alignment and capping rules.

The main reconciliation findings are:

- 6 severity policy IDs do not appear in the frequency table.
- 195 severity records are unmatched, representing about €788,714 of claim amount.
- 9,116 policies have positive claim counts but no matching severity record.
- 1 matched policy has a claim count that differs from the number of severity records.

## Modeling treatment

The modeling portfolio applies four rules:

- claim count capped at 4;
- exposure capped at 1 year;
- total policy claim amount capped at €200,000;
- positive claim counts with no recorded claim amount reset to zero for the aligned frequency/severity construction.

These changes are saved to `outputs/modeling_adjustments.csv`.

Large claims remain important after treatment. The top 1% of positive-loss policies account for 28.7% of modeled claim amount.

## Models

The GLM baseline uses a Poisson model for claim frequency and a Gamma model for conditional severity. Their product gives a two-part pure-premium estimate. A direct Tweedie GLM is included as a second actuarial benchmark.

A nonlinear challenger uses histogram gradient boosting with Poisson loss for frequency and Gamma loss for severity. This model can capture nonlinearities and interactions without manually adding interaction terms to the GLM.

## Model selection

The final test set is separated before tuning. Hyperparameters are chosen on the development sample. Model-family selection then uses three repeated validation splits and a common Tweedie deviance with power 1.5.

The validation results are:

- Boosted two-part: mean deviance 73.714 (SD 1.730), mean Gini 0.346.
- Two-part GLM: mean deviance 74.941 (SD 1.135), mean Gini 0.317.
- Direct Tweedie GLM: mean deviance 75.028 (SD 1.071), mean Gini 0.308.

Based on this rule, **Boosted two-part** is selected before the test set is evaluated.

## Test results

On the final test set, the selected model has:

- Tweedie deviance: **77.884**
- D²: **5.99%**
- Gini: **0.320**
- normalized Gini: **0.325**
- aggregate predicted / actual claim cost: **0.933**

For comparison, the two-part GLM test Gini is 0.278 and the direct Tweedie GLM test Gini is 0.282.

The boosted frequency component improves frequency D² relative to the Poisson GLM. Severity remains weak for both Gamma GLM and boosted severity, with slightly negative test D². This is consistent with the limited number of positive-loss policies and the concentration of claim cost in a small number of large losses.

## Risk segmentation

Risk groups are formed using **exposure-weighted deciles**, so each decile contains roughly the same amount of exposure rather than the same number of policies.

The highest-risk decile has observed pure premium €462.43, compared with €42.53 in the lowest-risk decile, a ratio of about **10.9×**.

The middle deciles are not perfectly monotonic. This is reported rather than smoothed away because claim severity is volatile and the test portfolio contains relatively few positive-loss observations.

## Relativities

Pricing relativities are calculated by changing one characteristic of a fixed reference policy and comparing the resulting prediction with the reference prediction. This provides interpretable model-implied relativities without treating regularized one-hot coefficients as traditional actuarial base-level factors.

## Limitations

The project is a pricing exercise rather than a production rate filing. Important limitations include:

- incomplete reconciliation between the frequency and severity sources;
- limited severity signal in the available rating variables;
- sensitivity of aggregate claim cost to large losses;
- no expense, profit, tax, capital or regulatory loading;
- no temporal validation because the public dataset does not provide a suitable policy-time structure for that purpose.

The GLM models are retained because they provide an interpretable actuarial benchmark even though the nonlinear challenger performs better on the chosen validation criterion.
