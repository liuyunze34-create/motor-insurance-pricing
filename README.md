# Motor Insurance Pricing

[![Tests](https://github.com/liuyunze34-create/motor-insurance-pricing/actions/workflows/tests.yml/badge.svg)](https://github.com/liuyunze34-create/motor-insurance-pricing/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modeling-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Actuarial motor insurance pricing using the French MTPL dataset. The project models claim frequency and severity separately, combines them into pure premium, and compares traditional GLMs with a nonlinear gradient-boosting challenger.

## At a glance

| Portfolio | Selected model | Test Gini | Test D² |
|---|---|---:|---:|
| **678,013 policies** | **Boosted two-part** | **0.320** | **5.99%** |

The highest exposure-weighted risk decile has about **10.9×** the observed pure premium of the lowest decile.

**Quick links:** [Dashboard](dashboard.html) · [Notebook](Motor_Insurance_Pricing.ipynb) · [Results](RESULTS.md) · [Portfolio summary](PORTFOLIO.md) · [Project report](PROJECT_REPORT.md)

## What the project does

The pricing target is pure premium:

`pure premium = expected claim frequency × expected claim severity`

Three approaches are compared:

1. **Two-part GLM** — Poisson frequency × Gamma severity.
2. **Direct Tweedie GLM** — pure premium modeled directly.
3. **Boosted two-part** — histogram gradient boosting for frequency and severity.

The final test set is split off before model-family selection. Hyperparameters are selected on development data, and the final model family is chosen from **three repeated validation splits** before the test set is evaluated.

## Validation results

Repeated-validation mean performance:

| Model | Mean Tweedie deviance | SD | Mean Gini |
|---|---:|---:|---:|
| **Boosted two-part** | **73.714** | 1.730 | **0.346** |
| Two-part GLM | 74.941 | 1.135 | 0.317 |
| Direct Tweedie GLM | 75.028 | 1.071 | 0.308 |

Final test performance for the selected boosted two-part model:

| Metric | Result |
|---|---:|
| Tweedie deviance | 77.884 |
| D² | 5.99% |
| Gini | 0.320 |
| Normalized Gini | 0.325 |
| Predicted / actual aggregate cost | 0.933 |
| Highest / lowest risk-decile pure premium | 10.9× |

Severity remains difficult to predict. The top 1% of positive-loss policies account for about **28.7%** of modeled claim amount even after the loss cap, so aggregate severity results remain sensitive to large losses. That limitation is kept visible rather than hidden.

## Data

The project uses the OpenML `freMTPL2freq` and `freMTPL2sev` datasets.

- Policy rows: **678,013**
- Raw exposure: **358,499 policy-years**
- Severity records: **26,639**
- Matched severity policy IDs: **24,944**
- Unmatched severity policy IDs: **6**
- Positive claim-count policies without a severity record: **9,116**

The raw and modeling portfolios are reported separately because the two source tables do not reconcile perfectly. The modeling treatment caps claim count at 4, exposure at 1 year, and policy claim amount at €200,000. Every adjustment is recorded in `outputs/modeling_adjustments.csv`.

After treatment, the modeling portfolio has observed frequency **0.0737** and observed pure premium **€145.58**.

## Pricing relativities

`outputs/pricing_relativities.csv` contains **prediction-based** relativities around a fixed reference policy. Each factor is changed one at a time while the rest of the policy is held constant. This avoids presenting regularized one-hot coefficients as if they were traditional base-level tariff relativities.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the two OpenML source tables:

```bash
python download_data.py
```

Run the full pipeline:

```bash
python run_project.py
```

Start the interactive pricing app:

```bash
streamlit run app.py
```

Run the checks:

```bash
pytest -q
```

## Repository structure

```text
Motor_Insurance_Pricing.ipynb   compact technical notebook
run_project.py                  reproducible modeling pipeline
project_utils.py                preparation, models and validation
app.py                          Streamlit policy pricing tool
dashboard.html                  static portfolio dashboard
download_data.py                OpenML data download script
outputs/                        compact validation/test summaries
tests/                          automated project checks
```

Raw CSVs, row-level test predictions, fitted `.joblib` files and generated PNG charts are intentionally excluded from version control. `download_data.py` and `run_project.py` recreate them.

## Scope

This is a portfolio pricing exercise, not a production tariff. Pure premium represents expected claim cost only; expenses, profit, taxes, capital requirements, underwriting rules and regulatory constraints are outside scope.
