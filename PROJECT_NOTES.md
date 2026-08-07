# Project Notes

## 30-second explanation

I built a motor insurance pricing project on about 678,000 French MTPL policies. I modeled frequency and severity separately using Poisson and Gamma GLMs, added a direct Tweedie benchmark, and then tested a nonlinear two-part gradient-boosting model. I kept the final test set untouched during model selection and used three repeated validation splits to choose the pricing model. The boosted two-part model was selected and achieved a test Gini of 0.320 and D² of 5.99%.

## Points to understand before using it on a resume

- Frequency is expected claims per exposure-year.
- Severity is expected claim cost conditional on a claim.
- Pure premium is expected claim cost per exposure-year.
- Poisson is used as an actuarial frequency benchmark.
- Gamma is used for positive, right-skewed conditional severity.
- Tweedie can model zero and positive pure premium directly.
- Gini measures ranking, not calibration.
- The final test set is not used to choose the model.
- Severity is the weak part of the project; both severity models have slightly negative test D².

## Results worth remembering

- Policies: 678,013
- Selected model: Boosted two-part
- Test D²: 5.99%
- Test Gini: 0.320
- Normalized Gini: 0.325
- High/low risk-decile observed pure-premium ratio: 10.9×

## Resume wording

**Motor Insurance Pricing & Risk Modeling | Python, pandas, scikit-learn, GLM, gradient boosting, Streamlit**

- Built an end-to-end pricing model on 678,000+ motor insurance policies, combining Poisson frequency and Gamma severity GLMs with direct Tweedie and gradient-boosting benchmarks.
- Used repeated validation, exposure-weighted calibration, Gini and risk-decile analysis to select a nonlinear two-part model, achieving 0.320 test Gini and 5.99% pure-premium D².
- Added source-data reconciliation, model-implied pricing relativities and an interactive policy-level pricing dashboard.

Use two bullets if space is tight. Do not describe the severity model as strong; the project is more credible when the limitation is explained directly.
