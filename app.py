from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
FEATURE_COLUMNS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]

st.set_page_config(page_title="Motor Insurance Pricing", page_icon="📊", layout="wide")


@st.cache_data
def load_outputs():
    names = {
        "summary": "portfolio_summary.csv",
        "quality": "data_quality_checks.csv",
        "reconciliation": "source_reconciliation.csv",
        "adjustments": "modeling_adjustments.csv",
        "metrics": "model_metrics.csv",
        "stability": "validation_stability_summary.csv",
        "deciles": "risk_deciles.csv",
        "age": "eda_driver_age.csv",
        "bonus": "eda_bonus_malus.csv",
        "claim_distribution": "eda_claim_distribution.csv",
        "relativities": "pricing_relativities.csv",
        "prediction_quantiles": "prediction_quantiles.csv",
    }
    return {key: pd.read_csv(OUTPUT_DIR / filename) for key, filename in names.items()}


@st.cache_resource
def load_models():
    required = [
        "preprocessor.joblib", "frequency_model.joblib", "severity_model.joblib",
        "tweedie_model.joblib", "boost_preprocessor.joblib",
        "boost_frequency_model.joblib", "boost_severity_model.joblib",
        "feature_defaults.json",
    ]
    if not all((MODEL_DIR / f).exists() for f in required):
        return None, None, None
    models = {
        "preprocessor": joblib.load(MODEL_DIR / "preprocessor.joblib"),
        "frequency": joblib.load(MODEL_DIR / "frequency_model.joblib"),
        "severity": joblib.load(MODEL_DIR / "severity_model.joblib"),
        "tweedie": joblib.load(MODEL_DIR / "tweedie_model.joblib"),
        "boost_preprocessor": joblib.load(MODEL_DIR / "boost_preprocessor.joblib"),
        "boost_frequency": joblib.load(MODEL_DIR / "boost_frequency_model.joblib"),
        "boost_severity": joblib.load(MODEL_DIR / "boost_severity_model.joblib"),
    }
    metadata = json.loads((OUTPUT_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    defaults = json.loads((MODEL_DIR / "feature_defaults.json").read_text(encoding="utf-8"))
    return models, metadata, defaults


def predict_policy(policy: pd.DataFrame, models: dict, metadata: dict):
    x = models["preprocessor"].transform(policy[FEATURE_COLUMNS])
    xb = models["boost_preprocessor"].transform(policy[FEATURE_COLUMNS])
    c = metadata["calibration_factors"]
    freq = float(models["frequency"].predict(x)[0] * c["frequency"])
    sev = float(models["severity"].predict(x)[0] * c["severity"])
    tw = float(models["tweedie"].predict(x)[0] * c["tweedie"])
    bf = float(models["boost_frequency"].predict(xb)[0] * c["boost_frequency"])
    bs = float(models["boost_severity"].predict(xb)[0] * c["boost_severity"])
    return {
        "Poisson frequency": freq,
        "Gamma severity": sev,
        "Two-part GLM": freq * sev,
        "Direct Tweedie GLM": tw,
        "Boosted frequency": bf,
        "Boosted severity": bs,
        "Boosted two-part": bf * bs,
    }


outputs = load_outputs()
models, metadata, defaults = load_models()
summary = outputs["summary"]
raw = summary.iloc[0]
modeling = summary.iloc[1]

st.title("Motor Insurance Pricing")
st.caption("French MTPL portfolio · frequency, severity and pure premium")
page = st.sidebar.radio("Page", ["Portfolio", "Models", "Pricing relativities", "Policy example"])

if page == "Portfolio":
    st.subheader("Portfolio")
    left, right = st.columns(2)
    for col, row, label in [(left, raw, "Raw joined portfolio"), (right, modeling, "Modeling portfolio")]:
        with col:
            st.markdown(f"**{label}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Policies", f"{int(row['Policies']):,}")
            c2.metric("Claims", f"{row['Claims']:,.0f}")
            c3.metric("Frequency", f"{row['ObservedFrequency']:.4f}")
            c4.metric("Pure premium", f"€{row['ObservedPurePremium']:,.2f}")

    st.caption("The modeling portfolio is shown separately because the frequency and severity sources do not reconcile perfectly and a small number of extreme observations are capped.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(outputs["claim_distribution"], x="ClaimCountDisplay", y="Policies", title="Policy distribution by claim count"), use_container_width=True)
    with right:
        st.plotly_chart(px.line(outputs["age"], x="DriverAgeBand", y="Frequency", markers=True, title="Observed frequency by driver age"), use_container_width=True)
    with st.expander("Source reconciliation"):
        st.dataframe(outputs["reconciliation"], use_container_width=True, hide_index=True)
    with st.expander("Modeling adjustments"):
        st.dataframe(outputs["adjustments"], use_container_width=True, hide_index=True)
    with st.expander("Data checks"):
        st.dataframe(outputs["quality"], use_container_width=True, hide_index=True)

elif page == "Models":
    st.subheader("Model comparison")
    selected = "Boosted two-part"
    if (OUTPUT_DIR / "model_metadata.json").exists():
        selected = json.loads((OUTPUT_DIR / "model_metadata.json").read_text(encoding="utf-8"))["selected_model"]
    st.write(f"The selected pricing model is **{selected}**. Model-family selection uses repeated validation; the final test set is held back until after that choice.")

    stability = outputs["stability"]
    st.plotly_chart(px.scatter(stability, x="Model", y="MeanValidationDeviance", error_y="SDValidationDeviance", title="Repeated validation", labels={"MeanValidationDeviance": "Mean Tweedie deviance (p=1.5)"}), use_container_width=True)
    st.dataframe(stability, use_container_width=True, hide_index=True)

    pricing = outputs["metrics"].loc[outputs["metrics"]["Target"] == "Pure premium"]
    st.markdown("**Final test results**")
    st.dataframe(pricing, use_container_width=True, hide_index=True)

    deciles = outputs["deciles"]
    fig = go.Figure()
    fig.add_bar(x=deciles["RiskDecile"], y=deciles["ObservedPurePremium"], name="Observed")
    fig.add_bar(x=deciles["RiskDecile"], y=deciles["PredictedPurePremium"], name="Predicted")
    fig.update_layout(barmode="group", title="Pure premium by exposure-weighted risk decile", xaxis_title="Risk decile", yaxis_title="Pure premium")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Severity remains noisy even after capping extreme claims, so the project reports ranking, deviance, calibration and repeated-validation stability rather than relying on one score.")

elif page == "Pricing relativities":
    st.subheader("Pricing relativities")
    st.write("Relativities are model-implied changes around a fixed reference policy. They are calculated from predictions rather than read directly from one-hot coefficients.")
    rel = outputs["relativities"].copy()
    variable = st.selectbox("Variable", rel["Variable"].drop_duplicates().tolist())
    subset = rel.loc[rel["Variable"] == variable].copy()
    subset["Label"] = subset["Level"].astype(str) + np.where(subset["IsReference"].astype(str).str.lower().eq("true"), " (reference)", "")
    fig = px.bar(subset, x="Label", y="SelectedModelRelativity", title=f"{variable} relativity")
    fig.add_hline(y=1.0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(subset.drop(columns=["Label"]), use_container_width=True, hide_index=True)

else:
    st.subheader("Policy example")
    if models is None:
        st.info("The fitted model files are generated locally rather than stored in the public repository. Run `python download_data.py` and `python run_project.py`, then restart Streamlit to enable the live policy example.")
        st.stop()

    pairs = pd.DataFrame(defaults["area_region_pairs"])
    c1, c2, c3 = st.columns(3)
    with c1:
        driv_age = st.number_input("Driver age", 18, 100, int(defaults["numeric"]["DrivAge"]))
        veh_age = st.number_input("Vehicle age", 0, 100, int(defaults["numeric"]["VehAge"]))
        veh_power = st.number_input("Vehicle power", 4, 15, int(defaults["numeric"]["VehPower"]))
        exposure = st.number_input("Exposure", 0.01, 1.00, 1.00, step=0.05)
    with c2:
        bonus_malus = st.number_input("Bonus-malus", 50, 230, int(defaults["numeric"]["BonusMalus"]))
        density = st.number_input("Population density", 1, 27000, int(defaults["numeric"]["Density"]))
        brand = st.selectbox("Vehicle brand", defaults["categories"]["VehBrand"])
        fuel = st.selectbox("Fuel", defaults["categories"]["VehGas"])
    with c3:
        regions = defaults["categories"]["Region"]
        region = st.selectbox("Region", regions)
        valid_areas = sorted(pairs.loc[pairs["Region"] == region, "Area"].astype(str).unique())
        area = st.selectbox("Area", valid_areas)
        methods = list(dict.fromkeys([metadata["selected_model"], "Two-part GLM", "Direct Tweedie GLM", "Boosted two-part"]))
        method = st.selectbox("Pricing method", methods)

    policy = pd.DataFrame([{
        "VehPower": int(veh_power), "VehAge": int(veh_age), "DrivAge": int(driv_age),
        "BonusMalus": int(bonus_malus), "VehBrand": brand, "VehGas": fuel,
        "Area": area, "Density": int(density), "Region": region,
    }])
    estimate = predict_policy(policy, models, metadata)
    premium = estimate[method]
    column = {"Two-part GLM": "PredictedPurePremiumTwoPartGLM", "Direct Tweedie GLM": "PredictedPurePremiumTweedie", "Boosted two-part": "PredictedPurePremiumBoosted"}[method]
    q = outputs["prediction_quantiles"].sort_values("Quantile")
    percentile = float(np.interp(premium, q[column], q["Quantile"] * 100, left=0, right=100))
    decile = int(np.clip(np.ceil(percentile / 10), 1, 10))
    portfolio_pp = float(modeling["ObservedPurePremium"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annual pure premium", f"€{premium:,.2f}")
    m2.metric("Expected cost for exposure", f"€{premium * exposure:,.2f}")
    m3.metric("Portfolio relativity", f"{premium / portfolio_pp:.2f}×")
    m4.metric("Risk percentile", f"{percentile:.0f}th (decile {decile})")
    st.markdown("**Frequency and severity components**")
    st.write({k: round(v, 4 if "frequency" in k.lower() else 2) for k, v in estimate.items() if k in ["Poisson frequency", "Gamma severity", "Boosted frequency", "Boosted severity"]})
    st.caption("Portfolio demonstration only; this is not a commercial insurance quote.")
