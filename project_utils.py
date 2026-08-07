from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TWEEDIE_EVAL_POWER = 1.5
STABILITY_SEEDS = [11, 23, 77]
FEATURE_COLUMNS = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Area", "Density", "Region",
]
CATEGORICAL_COLUMNS = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
NUMERIC_COLUMNS = ["VehAge", "DrivAge", "BonusMalus", "Density"]


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def weighted_mae(y_true, y_pred, weights) -> float:
    return weighted_mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), weights)


def load_raw_data(data_dir: Path):
    freq_path = data_dir / "freMTPL2freq.csv"
    sev_path = data_dir / "freMTPL2sev.csv"
    if not freq_path.exists() or not sev_path.exists():
        raise FileNotFoundError("Run download_data.py first to create the two OpenML CSV files.")
    freq = pd.read_csv(freq_path)
    sev = pd.read_csv(sev_path)
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev["IDpol"] = sev["IDpol"].astype(int)
    return freq, sev


def reconcile_sources(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    freq_ids = set(freq["IDpol"].unique())
    matched = sev[sev["IDpol"].isin(freq_ids)]
    unmatched = sev[~sev["IDpol"].isin(freq_ids)]
    sev_count = matched.groupby("IDpol").size().rename("SeverityRecords")
    joined = freq[["IDpol", "ClaimNb"]].merge(sev_count, left_on="IDpol", right_index=True, how="inner")
    no_sev = (freq["ClaimNb"] > 0) & (~freq["IDpol"].isin(matched["IDpol"]))
    rows = [
        ("Frequency policy rows", len(freq)),
        ("Frequency unique policy IDs", freq["IDpol"].nunique()),
        ("Severity claim records", len(sev)),
        ("Severity unique policy IDs", sev["IDpol"].nunique()),
        ("Matched severity policy IDs", matched["IDpol"].nunique()),
        ("Unmatched severity policy IDs", unmatched["IDpol"].nunique()),
        ("Unmatched severity records", len(unmatched)),
        ("Unmatched severity amount", float(unmatched["ClaimAmount"].sum())),
        ("Positive claim-count policies with no severity record", int(no_sev.sum())),
        ("Matched policies where claim count differs from severity record count", int((joined["ClaimNb"] != joined["SeverityRecords"]).sum())),
    ]
    return pd.DataFrame(rows, columns=["Check", "Value"])


def prepare_data(freq: pd.DataFrame, sev: pd.DataFrame):
    reconciliation = reconcile_sources(freq, sev)
    sev_by_policy = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    raw = freq.merge(sev_by_policy, on="IDpol", how="left")
    raw["ClaimAmount"] = raw["ClaimAmount"].fillna(0.0)
    for col in ["VehBrand", "VehGas", "Area", "Region"]:
        raw[col] = raw[col].astype(str).str.strip("'")
    raw["Frequency"] = raw["ClaimNb"] / raw["Exposure"]
    raw["AvgClaimAmount"] = raw["ClaimAmount"] / np.fmax(raw["ClaimNb"], 1)
    raw["PurePremium"] = raw["ClaimAmount"] / raw["Exposure"]

    quality = pd.DataFrame({
        "Check": ["Policy rows", "Unique policy IDs", "Duplicate policy IDs", "Missing values after join", "Exposure <= 0", "Exposure > 1", "Claim count > 4", "Claim amount > 200,000", "Positive claim count but zero recorded amount"],
        "Value": [len(raw), raw["IDpol"].nunique(), int(raw["IDpol"].duplicated().sum()), int(raw.isna().sum().sum()), int((raw["Exposure"] <= 0).sum()), int((raw["Exposure"] > 1).sum()), int((raw["ClaimNb"] > 4).sum()), int((raw["ClaimAmount"] > 200000).sum()), int(((raw["ClaimNb"] > 0) & (raw["ClaimAmount"] == 0)).sum())],
    })

    clean = raw.copy()
    reset_mask = (clean["ClaimNb"] > 0) & (clean["ClaimAmount"] == 0)
    adjustments = pd.DataFrame([
        ("Claim-count cap", "ClaimNb capped at 4", int((raw["ClaimNb"] > 4).sum()), float(raw["ClaimNb"].sum() - raw["ClaimNb"].clip(upper=4).sum())),
        ("Exposure cap", "Exposure capped at 1 year", int((raw["Exposure"] > 1).sum()), float(raw["Exposure"].sum() - raw["Exposure"].clip(upper=1).sum())),
        ("Loss cap", "Policy claim amount capped at €200,000", int((raw["ClaimAmount"] > 200000).sum()), float(raw["ClaimAmount"].sum() - raw["ClaimAmount"].clip(upper=200000).sum())),
        ("Count/severity alignment", "Positive claim count reset to 0 when no claim amount is recorded", int(reset_mask.sum()), float(raw.loc[reset_mask, "ClaimNb"].sum())),
    ], columns=["Adjustment", "Treatment", "AffectedPolicies", "AggregateChange"])

    clean["ClaimNb"] = clean["ClaimNb"].clip(upper=4)
    clean["Exposure"] = clean["Exposure"].clip(upper=1)
    clean["ClaimAmount"] = clean["ClaimAmount"].clip(upper=200000)
    clean.loc[reset_mask, "ClaimNb"] = 0
    clean["Frequency"] = clean["ClaimNb"] / clean["Exposure"]
    clean["AvgClaimAmount"] = clean["ClaimAmount"] / np.fmax(clean["ClaimNb"], 1)
    clean["PurePremium"] = clean["ClaimAmount"] / clean["Exposure"]
    return raw, clean, quality, reconciliation, adjustments


def portfolio_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    exposure = float(df["Exposure"].sum())
    claims = float(df["ClaimNb"].sum())
    amount = float(df["ClaimAmount"].sum())
    return pd.DataFrame([{
        "Portfolio": label, "Policies": len(df), "Exposure": exposure, "Claims": claims,
        "ClaimAmount": amount, "ObservedFrequency": claims / exposure,
        "ObservedSeverity": amount / claims if claims else np.nan,
        "ObservedPurePremium": amount / exposure,
    }])


def loss_concentration(clean: pd.DataFrame) -> pd.DataFrame:
    positive = clean.loc[clean["ClaimAmount"] > 0, "ClaimAmount"].sort_values(ascending=False)
    total = positive.sum()
    rows = []
    for share in [0.01, 0.05, 0.10]:
        n = max(1, int(np.ceil(len(positive) * share)))
        rows.append({"TopPolicyShare": share, "Policies": n, "ClaimAmountShare": float(positive.head(n).sum() / total)})
    return pd.DataFrame(rows)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("numeric", make_pipeline(StandardScaler()), NUMERIC_COLUMNS),
    ], sparse_threshold=1.0)


def make_boost_preprocessor():
    prep = ColumnTransformer([
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL_COLUMNS),
        ("numeric", "passthrough", NUMERIC_COLUMNS),
    ], sparse_threshold=0.0)
    mask = [True] * len(CATEGORICAL_COLUMNS) + [False] * len(NUMERIC_COLUMNS)
    return prep, mask


def concentration_curve(actual_amount, exposure, score):
    actual = np.asarray(actual_amount, dtype=float)
    exp = np.asarray(exposure, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score)
    actual, exp = actual[order], exp[order]
    x = np.r_[0.0, np.cumsum(exp) / exp.sum()]
    y = np.r_[0.0, np.cumsum(actual) / actual.sum()]
    gini = float(1.0 - 2.0 * np.trapz(y, x))
    return x, y, gini


def assign_exposure_groups(score: pd.Series, exposure: pd.Series, groups: int = 10) -> pd.Series:
    frame = pd.DataFrame({"score": score.to_numpy(), "exposure": exposure.to_numpy()}, index=score.index).sort_values("score")
    midpoint = frame["exposure"].cumsum() - frame["exposure"] / 2
    frame["group"] = np.minimum((midpoint / frame["exposure"].sum() * groups).astype(int) + 1, groups)
    return frame["group"].reindex(score.index)


def _model_metrics(y, pred, weights, power=TWEEDIE_EVAL_POWER):
    dev = mean_tweedie_deviance(y, np.fmax(pred, 1e-9), power=power, sample_weight=weights)
    base = np.repeat(weighted_mean(y, weights), len(y))
    null = mean_tweedie_deviance(y, base, power=power, sample_weight=weights)
    return dev, 1 - dev / null, weighted_mae(y, pred, weights)


def _fit_glm_models(train: pd.DataFrame, params: dict[str, float]):
    prep = make_preprocessor()
    x = prep.fit_transform(train[FEATURE_COLUMNS])
    freq = PoissonRegressor(alpha=params["frequency_alpha"], max_iter=400).fit(x, train["Frequency"], sample_weight=train["Exposure"])
    sev_mask = (train["ClaimNb"] > 0) & (train["ClaimAmount"] > 0)
    sev = GammaRegressor(alpha=params["severity_alpha"], max_iter=400).fit(x[sev_mask], train.loc[sev_mask, "AvgClaimAmount"], sample_weight=train.loc[sev_mask, "ClaimNb"])
    tweedie = TweedieRegressor(power=1.5, alpha=params["tweedie_alpha"], link="log", max_iter=400).fit(x, train["PurePremium"], sample_weight=train["Exposure"])
    return prep, freq, sev, tweedie


def _fit_boost_models(train: pd.DataFrame):
    prep, cat_mask = make_boost_preprocessor()
    xb = prep.fit_transform(train[FEATURE_COLUMNS])
    freq = HistGradientBoostingRegressor(loss="poisson", categorical_features=cat_mask, learning_rate=0.08, max_iter=180, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE).fit(xb, train["Frequency"], sample_weight=train["Exposure"])
    sev_mask = (train["ClaimNb"] > 0) & (train["ClaimAmount"] > 0)
    sev = HistGradientBoostingRegressor(loss="gamma", categorical_features=cat_mask, learning_rate=0.05, max_iter=140, max_leaf_nodes=10, l2_regularization=2.0, random_state=RANDOM_STATE).fit(xb[sev_mask], train.loc[sev_mask, "AvgClaimAmount"], sample_weight=train.loc[sev_mask, "ClaimNb"])
    return prep, freq, sev


def _calibration_factor(actual, predicted, weights):
    p = weighted_mean(predicted, weights)
    return weighted_mean(actual, weights) / p if p > 0 else 1.0


def tune_models(train: pd.DataFrame, validation: pd.DataFrame):
    candidate_alphas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    records = []
    prep = make_preprocessor()
    x_train = prep.fit_transform(train[FEATURE_COLUMNS])
    x_val = prep.transform(validation[FEATURE_COLUMNS])
    sev_train = (train["ClaimNb"] > 0) & (train["ClaimAmount"] > 0)
    sev_val = (validation["ClaimNb"] > 0) & (validation["ClaimAmount"] > 0)

    best = {}
    for name, cls, target, weight, mask, metric in [
        ("frequency", PoissonRegressor, "Frequency", "Exposure", None, mean_poisson_deviance),
        ("severity", GammaRegressor, "AvgClaimAmount", "ClaimNb", sev_train, mean_gamma_deviance),
    ]:
        scores = []
        for alpha in candidate_alphas:
            tr_x = x_train if mask is None else x_train[mask]
            tr = train if mask is None else train.loc[mask]
            model = cls(alpha=alpha, max_iter=400).fit(tr_x, tr[target], sample_weight=tr[weight])
            va = validation if name == "frequency" else validation.loc[sev_val]
            va_x = x_val if name == "frequency" else x_val[sev_val]
            pred = np.fmax(model.predict(va_x), 1e-9)
            score = metric(va[target], pred, sample_weight=va[weight])
            records.append({"Model": name, "Alpha": alpha, "ValidationDeviance": score})
            scores.append((score, alpha))
        best[f"{name}_alpha"] = min(scores)[1]

    tweedie_scores = []
    for alpha in candidate_alphas:
        model = TweedieRegressor(power=1.5, alpha=alpha, link="log", max_iter=400).fit(x_train, train["PurePremium"], sample_weight=train["Exposure"])
        pred = np.fmax(model.predict(x_val), 1e-9)
        score = mean_tweedie_deviance(validation["PurePremium"], pred, power=1.5, sample_weight=validation["Exposure"])
        records.append({"Model": "tweedie", "Alpha": alpha, "ValidationDeviance": score})
        tweedie_scores.append((score, alpha))
    best["tweedie_alpha"] = min(tweedie_scores)[1]
    return best, pd.DataFrame(records)


def tweedie_power_sensitivity(train: pd.DataFrame, validation: pd.DataFrame, alpha: float):
    prep = make_preprocessor()
    xt = prep.fit_transform(train[FEATURE_COLUMNS])
    xv = prep.transform(validation[FEATURE_COLUMNS])
    rows = []
    for power in [1.1, 1.3, 1.5, 1.7, 1.9]:
        model = TweedieRegressor(power=power, alpha=alpha, link="log", max_iter=400).fit(xt, train["PurePremium"], sample_weight=train["Exposure"])
        pred = np.fmax(model.predict(xv), 1e-9)
        dev = mean_tweedie_deviance(validation["PurePremium"], pred, power=power, sample_weight=validation["Exposure"])
        _, _, gini = concentration_curve(validation["ClaimAmount"], validation["Exposure"], pred)
        rows.append({"Power": power, "ValidationDevianceAtSamePower": dev, "Gini": gini})
    return pd.DataFrame(rows)


def _candidate_predictions(train: pd.DataFrame, validation: pd.DataFrame, params: dict[str, float]):
    prep, freq, sev, tw = _fit_glm_models(train, params)
    xv = prep.transform(validation[FEATURE_COLUMNS])
    train_x = prep.transform(train[FEATURE_COLUMNS])
    train_glm = freq.predict(train_x) * sev.predict(train_x)
    val_glm = freq.predict(xv) * sev.predict(xv)
    train_tw = tw.predict(train_x)
    val_tw = tw.predict(xv)

    bprep, bfreq, bsev = _fit_boost_models(train)
    bxt = bprep.transform(train[FEATURE_COLUMNS])
    bxv = bprep.transform(validation[FEATURE_COLUMNS])
    train_boost = bfreq.predict(bxt) * bsev.predict(bxt)
    val_boost = bfreq.predict(bxv) * bsev.predict(bxv)

    models = {
        "Two-part GLM": (train_glm, val_glm),
        "Direct Tweedie GLM": (train_tw, val_tw),
        "Boosted two-part": (train_boost, val_boost),
    }
    out = {}
    for name, (tr_pred, va_pred) in models.items():
        factor = _calibration_factor(train["PurePremium"], tr_pred, train["Exposure"])
        out[name] = np.fmax(va_pred * factor, 1e-9)
    return out


def validation_stability(development: pd.DataFrame, params: dict[str, float]):
    records = []
    for seed in STABILITY_SEEDS:
        train, val = train_test_split(development, test_size=0.20, random_state=seed)
        preds = _candidate_predictions(train, val, params)
        for name, pred in preds.items():
            dev, _, mae = _model_metrics(val["PurePremium"], pred, val["Exposure"])
            _, _, gini = concentration_curve(val["ClaimAmount"], val["Exposure"], pred)
            records.append({
                "SplitSeed": seed, "Model": name, "ValidationDeviance": dev,
                "Gini": gini, "CalibrationRatio": weighted_mean(pred, val["Exposure"]) / weighted_mean(val["PurePremium"], val["Exposure"]),
                "WeightedMAE": mae,
            })
    detail = pd.DataFrame(records)
    summary = detail.groupby("Model", as_index=False).agg(
        MeanValidationDeviance=("ValidationDeviance", "mean"), SDValidationDeviance=("ValidationDeviance", "std"),
        MeanGini=("Gini", "mean"), SDGini=("Gini", "std"),
        MeanCalibrationRatio=("CalibrationRatio", "mean"), SDCalibrationRatio=("CalibrationRatio", "std"),
        MeanWeightedMAE=("WeightedMAE", "mean"),
    ).sort_values("MeanValidationDeviance")
    selected = str(summary.iloc[0]["Model"])
    return detail, summary, selected


def _segment_calibration(test: pd.DataFrame, pred: np.ndarray, column: str, bins=None):
    frame = test[[column, "Exposure", "ClaimAmount"]].copy()
    frame["PredictedAmount"] = pred * frame["Exposure"]
    if bins is not None:
        frame["Group"] = pd.cut(frame[column], bins=bins, right=False, include_lowest=True).astype(str)
    else:
        frame["Group"] = frame[column].astype(str)
    g = frame.groupby("Group", observed=True).agg(Exposure=("Exposure", "sum"), ClaimAmount=("ClaimAmount", "sum"), PredictedAmount=("PredictedAmount", "sum"), Policies=("Exposure", "size")).reset_index()
    g["ObservedPurePremium"] = g["ClaimAmount"] / g["Exposure"]
    g["PredictedPurePremium"] = g["PredictedAmount"] / g["Exposure"]
    g["ActualPredictedRatio"] = g["ClaimAmount"] / np.fmax(g["PredictedAmount"], 1e-9)
    return g


def fit_final_models(development: pd.DataFrame, test: pd.DataFrame, params: dict[str, float], selected_model: str):
    prep, freq, sev, tw = _fit_glm_models(development, params)
    xt = prep.transform(development[FEATURE_COLUMNS])
    xv = prep.transform(test[FEATURE_COLUMNS])
    bprep, bfreq, bsev = _fit_boost_models(development)
    bxt = bprep.transform(development[FEATURE_COLUMNS])
    bxv = bprep.transform(test[FEATURE_COLUMNS])

    dev_preds = {
        "Two-part GLM": freq.predict(xt) * sev.predict(xt),
        "Direct Tweedie GLM": tw.predict(xt),
        "Boosted two-part": bfreq.predict(bxt) * bsev.predict(bxt),
    }
    test_preds = {
        "Two-part GLM": freq.predict(xv) * sev.predict(xv),
        "Direct Tweedie GLM": tw.predict(xv),
        "Boosted two-part": bfreq.predict(bxv) * bsev.predict(bxv),
    }
    factors = {name: _calibration_factor(development["PurePremium"], pred, development["Exposure"]) for name, pred in dev_preds.items()}
    for name in test_preds:
        test_preds[name] = np.fmax(test_preds[name] * factors[name], 1e-9)

    freq_factor = _calibration_factor(development["Frequency"], freq.predict(xt), development["Exposure"])
    sev_mask = (development["ClaimNb"] > 0) & (development["ClaimAmount"] > 0)
    sev_factor = _calibration_factor(development.loc[sev_mask, "AvgClaimAmount"], sev.predict(xt[sev_mask]), development.loc[sev_mask, "ClaimNb"])
    boost_freq_factor = _calibration_factor(development["Frequency"], bfreq.predict(bxt), development["Exposure"])
    boost_sev_factor = _calibration_factor(development.loc[sev_mask, "AvgClaimAmount"], bsev.predict(bxt[sev_mask]), development.loc[sev_mask, "ClaimNb"])

    rows = []
    oracle_x, oracle_y, oracle_gini = concentration_curve(test["ClaimAmount"], test["Exposure"], test["PurePremium"])
    for name, pred in test_preds.items():
        dev, d2, mae = _model_metrics(test["PurePremium"], pred, test["Exposure"])
        _, _, gini = concentration_curve(test["ClaimAmount"], test["Exposure"], pred)
        rows.append({"Target": "Pure premium", "Model": name, "Selected": name == selected_model, "TestDeviance": dev, "D2Explained": d2, "WeightedMAE": mae, "CalibrationRatio": weighted_mean(pred, test["Exposure"]) / weighted_mean(test["PurePremium"], test["Exposure"]), "Gini": gini, "NormalizedGini": gini / oracle_gini})

    freq_pred = np.fmax(freq.predict(xv) * freq_factor, 1e-9)
    fdev = mean_poisson_deviance(test["Frequency"], freq_pred, sample_weight=test["Exposure"])
    fbase = np.repeat(weighted_mean(test["Frequency"], test["Exposure"]), len(test))
    fnull = mean_poisson_deviance(test["Frequency"], fbase, sample_weight=test["Exposure"])
    rows.append({"Target": "Frequency", "Model": "Poisson GLM", "Selected": False, "TestDeviance": fdev, "D2Explained": 1-fdev/fnull, "WeightedMAE": weighted_mae(test["Frequency"], freq_pred, test["Exposure"]), "CalibrationRatio": weighted_mean(freq_pred, test["Exposure"]) / weighted_mean(test["Frequency"], test["Exposure"]), "Gini": np.nan, "NormalizedGini": np.nan})

    test_sev = (test["ClaimNb"] > 0) & (test["ClaimAmount"] > 0)
    sev_pred = np.fmax(sev.predict(xv[test_sev]) * sev_factor, 1e-9)
    sdev = mean_gamma_deviance(test.loc[test_sev, "AvgClaimAmount"], sev_pred, sample_weight=test.loc[test_sev, "ClaimNb"])
    sbase = np.repeat(weighted_mean(test.loc[test_sev, "AvgClaimAmount"], test.loc[test_sev, "ClaimNb"]), test_sev.sum())
    snull = mean_gamma_deviance(test.loc[test_sev, "AvgClaimAmount"], sbase, sample_weight=test.loc[test_sev, "ClaimNb"])
    rows.append({"Target": "Severity", "Model": "Gamma GLM", "Selected": False, "TestDeviance": sdev, "D2Explained": 1-sdev/snull, "WeightedMAE": weighted_mae(test.loc[test_sev, "AvgClaimAmount"], sev_pred, test.loc[test_sev, "ClaimNb"]), "CalibrationRatio": weighted_mean(sev_pred, test.loc[test_sev, "ClaimNb"]) / weighted_mean(test.loc[test_sev, "AvgClaimAmount"], test.loc[test_sev, "ClaimNb"]), "Gini": np.nan, "NormalizedGini": np.nan})

    pred_frame = test[["IDpol", "Exposure", "ClaimNb", "ClaimAmount", "PurePremium"]].copy()
    pred_frame["PredictedPurePremiumTwoPartGLM"] = test_preds["Two-part GLM"]
    pred_frame["PredictedPurePremiumTweedie"] = test_preds["Direct Tweedie GLM"]
    pred_frame["PredictedPurePremiumBoosted"] = test_preds["Boosted two-part"]
    selected_col = {"Two-part GLM": "PredictedPurePremiumTwoPartGLM", "Direct Tweedie GLM": "PredictedPurePremiumTweedie", "Boosted two-part": "PredictedPurePremiumBoosted"}[selected_model]
    pred_frame["SelectedPredictedPurePremium"] = pred_frame[selected_col]

    decile = assign_exposure_groups(pred_frame["SelectedPredictedPurePremium"], pred_frame["Exposure"], 10)
    pred_frame["RiskDecile"] = decile
    d = pred_frame.groupby("RiskDecile").agg(Exposure=("Exposure", "sum"), ClaimAmount=("ClaimAmount", "sum"), PredictedAmount=("SelectedPredictedPurePremium", lambda s: float((s * pred_frame.loc[s.index, "Exposure"]).sum())), Policies=("IDpol", "size")).reset_index()
    d["ObservedPurePremium"] = d["ClaimAmount"] / d["Exposure"]
    d["PredictedPurePremium"] = d["PredictedAmount"] / d["Exposure"]
    d["ActualPredictedRatio"] = d["ClaimAmount"] / np.fmax(d["PredictedAmount"], 1e-9)

    selected_pred = pred_frame["SelectedPredictedPurePremium"].to_numpy()
    _, _, selected_gini = concentration_curve(test["ClaimAmount"], test["Exposure"], selected_pred)
    return {
        "preprocessor": prep, "frequency_model": freq, "severity_model": sev, "tweedie_model": tw,
        "boost_preprocessor": bprep, "boost_frequency_model": bfreq, "boost_severity_model": bsev,
        "selected_model": selected_model, "metrics": pd.DataFrame(rows), "predictions": pred_frame, "deciles": d,
        "selected_model_gini": selected_gini, "oracle_gini": oracle_gini, "selected_normalized_gini": selected_gini / oracle_gini,
        "calibration_factors": {"frequency": freq_factor, "severity": sev_factor, "tweedie": factors["Direct Tweedie GLM"], "boost_frequency": boost_freq_factor, "boost_severity": boost_sev_factor, "two_part": factors["Two-part GLM"], "boost_two_part": factors["Boosted two-part"]},
        "segment_calibration": {
            "driver_age": _segment_calibration(test, selected_pred, "DrivAge", bins=[18,25,30,40,50,60,70,80,101]),
            "bonus_malus": _segment_calibration(test, selected_pred, "BonusMalus", bins=[50,60,75,100,125,150,200,231]),
            "area": _segment_calibration(test, selected_pred, "Area"),
        },
    }


def create_eda_tables(clean: pd.DataFrame):
    claim = clean["ClaimNb"].clip(upper=5).value_counts().sort_index().rename_axis("ClaimCountDisplay").reset_index(name="Policies")
    claim["ClaimCountDisplay"] = claim["ClaimCountDisplay"].astype(str)

    def grouped(column, bins=None, name=None):
        frame = clean.copy()
        key = column
        if bins is not None:
            key = name or f"{column}Band"
            frame[key] = pd.cut(frame[column], bins=bins, right=False, include_lowest=True).astype(str)
        g = frame.groupby(key, observed=True).agg(Policies=("IDpol", "size"), Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum"), ClaimAmount=("ClaimAmount", "sum")).reset_index()
        g["Frequency"] = g["Claims"] / g["Exposure"]
        g["PurePremium"] = g["ClaimAmount"] / g["Exposure"]
        return g

    return {
        "claim_distribution": claim,
        "driver_age": grouped("DrivAge", [18,25,30,40,50,60,70,80,101], "DriverAgeBand"),
        "vehicle_age": grouped("VehAge", [0,2,5,10,15,20,101], "VehicleAgeBand"),
        "bonus_malus": grouped("BonusMalus", [50,60,75,100,125,150,200,231], "BonusMalusBand"),
        "area": grouped("Area"), "brand": grouped("VehBrand"), "fuel": grouped("VehGas"),
    }


def reference_policy(clean: pd.DataFrame) -> dict[str, Any]:
    return {
        "VehPower": int(clean["VehPower"].median()), "VehAge": int(clean["VehAge"].median()),
        "DrivAge": int(clean["DrivAge"].median()), "BonusMalus": int(clean["BonusMalus"].median()),
        "VehBrand": str(clean["VehBrand"].mode().iloc[0]), "VehGas": str(clean["VehGas"].mode().iloc[0]),
        "Area": str(clean["Area"].mode().iloc[0]), "Density": int(clean["Density"].median()),
        "Region": str(clean["Region"].mode().iloc[0]),
    }


def _predict_rows(frame: pd.DataFrame, result: dict[str, Any]):
    x = result["preprocessor"].transform(frame[FEATURE_COLUMNS])
    xb = result["boost_preprocessor"].transform(frame[FEATURE_COLUMNS])
    cal = result["calibration_factors"]
    gl = result["frequency_model"].predict(x) * result["severity_model"].predict(x) * cal["two_part"]
    tw = result["tweedie_model"].predict(x) * cal["tweedie"]
    bo = result["boost_frequency_model"].predict(xb) * result["boost_severity_model"].predict(xb) * cal["boost_two_part"]
    return gl, tw, bo


def prediction_relativities(clean: pd.DataFrame, result: dict[str, Any]):
    ref = reference_policy(clean)
    specs = {
        "Driver age": ("DrivAge", [20,25,30,40,50,60,70,80]),
        "Vehicle age": ("VehAge", [0,2,5,10,15,20]),
        "Bonus-malus": ("BonusMalus", [50,60,75,100,125,150,200]),
        "Area": ("Area", sorted(clean["Area"].astype(str).unique())),
        "Fuel": ("VehGas", sorted(clean["VehGas"].astype(str).unique())),
        "Vehicle power": ("VehPower", sorted(clean["VehPower"].astype(int).unique())),
    }
    ref_frame = pd.DataFrame([ref])
    rgl, rtw, rbo = [x[0] for x in _predict_rows(ref_frame, result)]
    rows = []
    for variable, (column, levels) in specs.items():
        for level in levels:
            row = ref.copy(); row[column] = level
            gl, tw, bo = [x[0] for x in _predict_rows(pd.DataFrame([row]), result)]
            rows.append({
                "Variable": variable, "Level": str(level), "ReferenceLevel": str(ref[column]), "IsReference": str(level) == str(ref[column]),
                "GLMTwoPartRelativity": gl / rgl, "TweedieRelativity": tw / rtw, "BoostedRelativity": bo / rbo,
                "SelectedModelRelativity": {"Two-part GLM": gl/rgl, "Direct Tweedie GLM": tw/rtw, "Boosted two-part": bo/rbo}[result["selected_model"]],
            })
    return pd.DataFrame(rows)


def save_models(result: dict[str, Any], model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "preprocessor": "preprocessor.joblib", "frequency_model": "frequency_model.joblib", "severity_model": "severity_model.joblib",
        "tweedie_model": "tweedie_model.joblib", "boost_preprocessor": "boost_preprocessor.joblib",
        "boost_frequency_model": "boost_frequency_model.joblib", "boost_severity_model": "boost_severity_model.joblib",
    }
    for key, filename in mapping.items():
        joblib.dump(result[key], model_dir / filename)
    (model_dir / "selected_model.txt").write_text(result["selected_model"], encoding="utf-8")


def save_defaults(clean: pd.DataFrame, output_path: Path):
    ref = reference_policy(clean)
    pairs = clean[["Region", "Area"]].drop_duplicates().sort_values(["Region", "Area"])
    payload = {
        "numeric": {"VehPower": int(ref["VehPower"]), "VehAge": int(ref["VehAge"]), "DrivAge": int(ref["DrivAge"]), "BonusMalus": int(ref["BonusMalus"]), "Density": int(ref["Density"]), "Exposure": 1.0},
        "categories": {c: sorted(clean[c].astype(str).unique().tolist()) for c in ["VehBrand", "VehGas", "Area", "Region"]},
        "reference_policy": ref, "area_region_pairs": pairs.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_matplotlib_figures(figures_dir: Path, eda_tables: dict[str, pd.DataFrame], result: dict[str, Any], relativities: pd.DataFrame, stability: pd.DataFrame):
    figures_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("01_claim_count_distribution.png", eda_tables["claim_distribution"], "ClaimCountDisplay", "Policies", "Policy distribution by claim count"),
        ("02_frequency_by_driver_age.png", eda_tables["driver_age"], "DriverAgeBand", "Frequency", "Observed frequency by driver age"),
        ("03_frequency_by_bonus_malus.png", eda_tables["bonus_malus"], "BonusMalusBand", "Frequency", "Observed frequency by bonus-malus"),
    ]
    for filename, table, x, y, title in plots:
        plt.figure(figsize=(9,5)); plt.bar(table[x].astype(str), table[y]); plt.title(title); plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(figures_dir/filename, dpi=150); plt.close()
    d = result["deciles"]
    plt.figure(figsize=(9,5)); plt.plot(d["RiskDecile"], d["ObservedPurePremium"], marker="o", label="Observed"); plt.plot(d["RiskDecile"], d["PredictedPurePremium"], marker="o", label="Predicted"); plt.legend(); plt.title("Pure premium by risk decile"); plt.tight_layout(); plt.savefig(figures_dir/"04_decile_calibration.png", dpi=150); plt.close()
    pred = result["predictions"]
    x,y,_ = concentration_curve(pred["ClaimAmount"], pred["Exposure"], pred["SelectedPredictedPurePremium"])
    plt.figure(figsize=(7,7)); plt.plot(x,y,label="Selected model"); plt.plot([0,1],[0,1],"--",label="Random"); plt.legend(); plt.title("Ordered Lorenz curve"); plt.tight_layout(); plt.savefig(figures_dir/"05_ordered_lorenz_curve.png", dpi=150); plt.close()
    rel = relativities[relativities["Variable"].isin(["Driver age", "Bonus-malus", "Area"])].copy(); rel["Label"] = rel["Variable"] + ": " + rel["Level"]
    show = pd.concat([rel.nsmallest(7,"SelectedModelRelativity"), rel.nlargest(7,"SelectedModelRelativity")]).drop_duplicates("Label").sort_values("SelectedModelRelativity")
    plt.figure(figsize=(9,6)); plt.barh(show["Label"], show["SelectedModelRelativity"]); plt.axvline(1, linestyle="--"); plt.title("Selected model relativities"); plt.tight_layout(); plt.savefig(figures_dir/"06_pricing_relativities.png", dpi=150); plt.close()
    s = stability.sort_values("MeanValidationDeviance")
    plt.figure(figsize=(9,5)); plt.errorbar(np.arange(len(s)), s["MeanValidationDeviance"], yerr=s["SDValidationDeviance"], fmt="o", capsize=5); plt.xticks(np.arange(len(s)), s["Model"], rotation=15); plt.title("Repeated validation"); plt.tight_layout(); plt.savefig(figures_dir/"07_validation_stability.png", dpi=150); plt.close()


def create_dashboard_html(output_path: Path, raw_summary: pd.DataFrame, clean_summary: pd.DataFrame, reconciliation: pd.DataFrame, adjustments: pd.DataFrame, metrics: pd.DataFrame, eda_tables: dict[str, pd.DataFrame], result: dict[str, Any], relativities: pd.DataFrame, stability_summary: pd.DataFrame, power_sensitivity: pd.DataFrame):
    raw, model = raw_summary.iloc[0], clean_summary.iloc[0]
    pricing = metrics[metrics["Target"] == "Pure premium"].copy()
    ratio = result["deciles"].iloc[-1]["ObservedPurePremium"] / result["deciles"].iloc[0]["ObservedPurePremium"]
    css = "body{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#222}h1,h2{margin-bottom:8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{border:1px solid #ddd;border-radius:8px;padding:14px}.v{font-size:24px;font-weight:700}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #ddd;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.section{margin:28px 0}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}}"
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Motor Insurance Pricing</title><style>{css}</style></head><body>
    <h1>Motor Insurance Pricing</h1><p>French MTPL frequency, severity and pure-premium modeling.</p>
    <div class="section"><h2>Modeling portfolio</h2><div class="cards"><div class="card">Policies<div class="v">{int(model['Policies']):,}</div></div><div class="card">Exposure<div class="v">{model['Exposure']:,.0f}</div></div><div class="card">Frequency<div class="v">{model['ObservedFrequency']:.4f}</div></div><div class="card">Pure premium<div class="v">€{model['ObservedPurePremium']:,.2f}</div></div></div></div>
    <div class="section"><h2>Model comparison</h2><p>Selected through repeated validation: <b>{result['selected_model']}</b>. Final test Gini: <b>{result['selected_model_gini']:.3f}</b>. Highest/lowest risk-decile observed pure-premium ratio: <b>{ratio:.1f}×</b>.</p>{stability_summary.round(4).to_html(index=False)}<h3>Final test</h3>{pricing.round(4).to_html(index=False)}</div>
    <div class="section"><h2>Risk deciles</h2>{result['deciles'].round(2).to_html(index=False)}</div>
    <div class="section"><h2>Source reconciliation</h2>{reconciliation.to_html(index=False)}<h3>Modeling adjustments</h3>{adjustments.round(2).to_html(index=False)}</div>
    <div class="section"><h2>Pricing relativities</h2>{relativities[['Variable','Level','ReferenceLevel','SelectedModelRelativity']].round(3).to_html(index=False)}</div>
    <div class="section"><h2>Tweedie power sensitivity</h2>{power_sensitivity.round(4).to_html(index=False)}</div>
    <p><small>Portfolio exercise only. Pure premium excludes expenses, profit, tax, capital and regulatory adjustments.</small></p></body></html>'''
    output_path.write_text(html, encoding="utf-8")
