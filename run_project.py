from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import project_utils as project_utils_module
from project_utils import (
    RANDOM_STATE,
    create_dashboard_html,
    create_eda_tables,
    fit_final_models,
    load_raw_data,
    loss_concentration,
    portfolio_summary,
    prediction_relativities,
    prepare_data,
    save_defaults,
    save_matplotlib_figures,
    save_models,
    tune_models,
    tweedie_power_sensitivity,
    validation_stability,
)

# Keep the GLM design matrix dense. This avoids SciPy sparse-indexing issues
# when severity observations are selected with a pandas Boolean mask.
_original_make_preprocessor = project_utils_module.make_preprocessor

def _dense_make_preprocessor():
    preprocessor = _original_make_preprocessor()
    preprocessor.set_params(sparse_threshold=0.0)
    return preprocessor

project_utils_module.make_preprocessor = _dense_make_preprocessor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    freq, sev = load_raw_data(DATA_DIR)
    raw, clean, data_quality, reconciliation, adjustments = prepare_data(freq, sev)

    raw_summary = portfolio_summary(raw, "Raw joined portfolio")
    clean_summary = portfolio_summary(clean, "Modeling portfolio")
    summary = pd.concat([raw_summary, clean_summary], ignore_index=True)
    concentration = loss_concentration(clean)

    # Final test set is separated before tuning or model-family selection.
    development, test = train_test_split(clean, test_size=0.20, random_state=RANDOM_STATE)
    tuning_train, tuning_validation = train_test_split(
        development, test_size=0.20, random_state=RANDOM_STATE
    )

    selected_params, tuning = tune_models(tuning_train, tuning_validation)
    power_sensitivity = tweedie_power_sensitivity(
        tuning_train, tuning_validation, selected_params["tweedie_alpha"]
    )

    stability_detail, stability_summary, selected_model = validation_stability(
        development, selected_params
    )

    result = fit_final_models(development, test, selected_params, selected_model)
    eda_tables = create_eda_tables(clean)
    relativities = prediction_relativities(clean, result)

    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    data_quality.to_csv(OUTPUT_DIR / "data_quality_checks.csv", index=False)
    reconciliation.to_csv(OUTPUT_DIR / "source_reconciliation.csv", index=False)
    adjustments.to_csv(OUTPUT_DIR / "modeling_adjustments.csv", index=False)
    concentration.to_csv(OUTPUT_DIR / "loss_concentration.csv", index=False)
    tuning.to_csv(OUTPUT_DIR / "hyperparameter_tuning.csv", index=False)
    power_sensitivity.to_csv(OUTPUT_DIR / "tweedie_power_sensitivity.csv", index=False)
    stability_detail.to_csv(OUTPUT_DIR / "validation_stability.csv", index=False)
    stability_summary.to_csv(OUTPUT_DIR / "validation_stability_summary.csv", index=False)
    result["metrics"].to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    result["deciles"].to_csv(OUTPUT_DIR / "risk_deciles.csv", index=False)
    relativities.to_csv(OUTPUT_DIR / "pricing_relativities.csv", index=False)
    result["predictions"].to_csv(
        OUTPUT_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
    )

    # Compact reference distribution used by the Streamlit policy example.
    # This avoids committing the full row-level test prediction file to GitHub.
    quantile_levels = [i / 100 for i in range(101)]
    prediction_quantiles = pd.DataFrame({
        "Quantile": quantile_levels,
        "PredictedPurePremiumTwoPartGLM": result["predictions"]["PredictedPurePremiumTwoPartGLM"].quantile(quantile_levels).to_numpy(),
        "PredictedPurePremiumTweedie": result["predictions"]["PredictedPurePremiumTweedie"].quantile(quantile_levels).to_numpy(),
        "PredictedPurePremiumBoosted": result["predictions"]["PredictedPurePremiumBoosted"].quantile(quantile_levels).to_numpy(),
    })
    prediction_quantiles.to_csv(OUTPUT_DIR / "prediction_quantiles.csv", index=False)

    for name, table in eda_tables.items():
        table.to_csv(OUTPUT_DIR / f"eda_{name}.csv", index=False)
    for name, table in result["segment_calibration"].items():
        table.to_csv(OUTPUT_DIR / f"calibration_{name}.csv", index=False)

    metadata = {
        "selected_params": selected_params,
        "selected_model": selected_model,
        "selection_rule": "Lowest mean Tweedie deviance across three repeated validation splits",
        "selected_model_gini": result["selected_model_gini"],
        "oracle_gini": result["oracle_gini"],
        "selected_normalized_gini": result["selected_normalized_gini"],
        "calibration_factors": result["calibration_factors"],
        "random_state": RANDOM_STATE,
        "test_fraction": 0.20,
        "stability_seeds": stability_detail["SplitSeed"].drop_duplicates().tolist(),
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    save_models(result, MODEL_DIR)
    save_defaults(clean, MODEL_DIR / "feature_defaults.json")
    save_matplotlib_figures(FIGURES_DIR, eda_tables, result, relativities, stability_summary)
    create_dashboard_html(
        ROOT / "dashboard.html",
        raw_summary,
        clean_summary,
        reconciliation,
        adjustments,
        result["metrics"],
        eda_tables,
        result,
        relativities,
        stability_summary,
        power_sensitivity,
    )

    print("Project run completed.")
    print("Selected parameters:", selected_params)
    print("Selected pricing model:", selected_model)
    print("\nRepeated validation:")
    print(stability_summary.to_string(index=False))
    print("\nFinal test metrics:")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
