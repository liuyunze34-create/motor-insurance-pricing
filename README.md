# Motor Insurance Pricing

An actuarial pricing project using the French motor third-party liability (MTPL) data. The project models claim frequency and claim severity separately, combines them into pure premium, and compares that approach with a direct Tweedie model and a nonlinear two-part model.

The modeling setup follows the standard French MTPL frequency/severity case study. This version extends it with source-data reconciliation, repeated validation, explicit model-family selection, a gradient-boosting challenger, prediction-based pricing relativities, segment calibration, and a policy-level dashboard.

## Data

The project uses the OpenML `freMTPL2freq` and `freMTPL2sev` datasets.

- Policies: **678,013**
- Raw exposure: **358,499 policy-years**
- Severity records: **26,639**
- Matched severity policy IDs: **24,944**
- Unmatched severity policy IDs: **6**
- Positive claim-count policies without a severity record: **9,116**

The raw and modeling portfolios are shown separately because the two source tables do not reconcile perfectly. The modeling treatment also caps claim count at 4, exposure at 1 year, and policy claim amount at €200,000. These adjustments are recorded in `outputs/modeling_adjustments.csv` rather than hidden in preprocessing.

After treatment, the modeling portfolio has observed frequency **0.0737** and observed pure premium **€145.58**.

## Models

Three pricing approaches are compared:

1. **Two-part GLM** — Poisson frequency × Gamma severity.
2. **Direct Tweedie GLM** — pure premium modeled directly with Tweedie power 1.5.
3. **Boosted two-part** — histogram gradient boosting for frequency and severity.

Exposure is used as the weight for frequency and pure-premium models. Claim count is used as the weight for conditional severity.

## Validation

The final test set is split off before tuning. Hyperparameters are chosen on the development sample, then the pricing model family is selected using **three repeated validation splits**. The test set is used only after the model family has been fixed.

Repeated-validation mean Tweedie deviance:

| Model | Mean deviance | SD | Mean Gini |
|---|---:|---:|---:|
| Boosted two-part | 73.714 | 1.730 | 0.346 |
| Two-part GLM | 74.941 | 1.135 | 0.317 |
| Direct Tweedie GLM | 75.028 | 1.071 | 0.308 |

The selected model is **Boosted two-part**.

## Test results

| Metric | Selected model |
|---|---:|
| Tweedie deviance | 77.884 |
| D² | 5.99% |
| Gini | 0.320 |
| Normalized Gini | 0.325 |
| Predicted / actual aggregate cost | 0.933 |
| Highest / lowest observed risk-decile pure premium | 10.9× |

The selected model improves risk ranking and deviance relative to the GLM alternatives, but severity remains difficult to predict. The top 1% of positive-loss policies account for about **28.7%** of modeled claim amount even after the loss cap, so severity and aggregate calibration remain sensitive to large losses.

## Pricing relativities

`outputs/pricing_relativities.csv` reports prediction-based relativities around a fixed reference policy. These are not raw one-hot coefficients. For each variable, one characteristic is changed while the rest of the reference profile is held fixed.

This makes the output easier to interpret as a pricing comparison and avoids presenting regularized one-hot coefficients as traditional actuarial base-level relativities.

## Run the project

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Raw CSV files are not committed to the public repository. Download them from OpenML with:

```bash
python download_data.py
```

Run the full pipeline:

```bash
python run_project.py
```

Open the static report in `dashboard.html`, or start the interactive app:

```bash
streamlit run app.py
```

Run the checks:

```bash
pytest -q
```

## Reproducing the results

The public repository keeps the raw CSVs, row-level test predictions, fitted `.joblib` objects and generated PNG charts out of version control. `download_data.py` restores the source data, and `run_project.py` recreates the detailed outputs, figures and fitted models. Compact validation and test summaries are committed so the main results can be reviewed without rerunning the full pipeline.

## Project files

```text
Motor_Insurance_Pricing.ipynb   notebook summary
run_project.py                  reproducible modeling pipeline
project_utils.py                preprocessing, models and validation
app.py                          Streamlit dashboard and policy example
dashboard.html                  static dashboard
download_data.py                OpenML download script
outputs/                        compact validation and test summaries
figures/                        generated chart documentation
models/                         generated model documentation
tests/                          data and pipeline checks
```

## Notes

This is a portfolio pricing exercise, not a production tariff. Pure premium represents expected claim cost only; expenses, profit, tax, capital cost and regulatory constraints are outside scope.

The final model should be described as a **nonlinear challenger selected through repeated validation**, with the GLMs retained as interpretable actuarial benchmarks. The negative severity D² is kept in the results because it is an important limitation of the available features rather than something to hide.
