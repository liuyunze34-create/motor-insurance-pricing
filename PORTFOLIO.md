# Portfolio summary

## One-line description

Motor insurance pricing project on 678,000+ policies using frequency-severity modeling, GLMs and gradient boosting, with repeated validation and a policy-level pricing dashboard.

## Resume version

**Motor Insurance Pricing & Risk Modeling** | Python, pandas, scikit-learn, GLM, Gradient Boosting

- Built an end-to-end motor insurance pricing model on **678,000+ policies**, modeling claim frequency and severity using **Poisson/Gamma GLMs, Tweedie GLM and gradient boosting** to estimate policy-level pure premium.
- Developed a validation framework using repeated train/validation splits, exposure-weighted calibration and Gini analysis; selected model achieved **0.320 test Gini** and approximately **10.9× observed pure-premium separation** between the highest- and lowest-risk deciles.

## 30-second explanation

I built a motor insurance pricing project using about 678,000 policies. I modeled claim frequency and claim severity separately and combined them into pure premium. I compared traditional actuarial GLMs with a nonlinear gradient-boosting challenger, selected the model family using repeated validation rather than the final test set, and then evaluated the selected model using deviance, calibration, risk deciles and Gini. The final model achieved a test Gini of about 0.32. I also built a simple dashboard to show policy-level pricing predictions and risk relativities.

## Technical interview version

The pricing target is pure premium, defined as expected claim frequency multiplied by expected claim severity. Frequency is modeled per unit exposure; conditional severity is modeled on positive-loss policies with claim count used as the weight. The GLM benchmarks are Poisson frequency, Gamma severity and direct Tweedie pure premium. A histogram gradient-boosting frequency/severity model is included as a nonlinear challenger. Model-family selection uses repeated validation splits, while the final test set is reserved for evaluation. The project reports Tweedie deviance, D², Gini, normalized Gini, exposure-weighted calibration and risk-decile results.

## What to show a recruiter

1. The README for the project summary and results.
2. `dashboard.html` for a quick visual overview.
3. `Motor_Insurance_Pricing.ipynb` for a concise technical walkthrough.
4. `PROJECT_REPORT.md` only if a technical interviewer wants more detail.

The repository is a portfolio exercise rather than a production insurance tariff. Expenses, profit, capital, regulation and production model governance are outside scope.
