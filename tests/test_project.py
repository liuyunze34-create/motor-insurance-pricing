from pathlib import Path

import numpy as np
import pandas as pd

from project_utils import assign_exposure_groups, load_raw_data, portfolio_summary, prepare_data

ROOT = Path(__file__).resolve().parents[1]


def test_data_reconciliation_and_caps():
    freq, sev = load_raw_data(ROOT / "data")
    raw, clean, quality, reconciliation, adjustments = prepare_data(freq, sev)

    assert len(raw) == 678_013
    assert raw["IDpol"].nunique() == len(raw)
    assert clean["Exposure"].max() <= 1
    assert clean["ClaimNb"].max() <= 4
    assert clean["ClaimAmount"].max() <= 200_000
    assert int(reconciliation.loc[reconciliation["Check"] == "Unmatched severity policy IDs", "Value"].iloc[0]) == 6
    assert int(reconciliation.loc[reconciliation["Check"] == "Positive claim-count policies with no severity record", "Value"].iloc[0]) == 9_116
    assert len(adjustments) == 4


def test_pure_premium_identity():
    freq, sev = load_raw_data(ROOT / "data")
    _, clean, *_ = prepare_data(freq, sev)
    summary = portfolio_summary(clean, "modeling").iloc[0]
    assert np.isclose(
        summary["ObservedFrequency"] * summary["ObservedSeverity"],
        summary["ObservedPurePremium"],
    )
    assert np.allclose(clean["PurePremium"], clean["ClaimAmount"] / clean["Exposure"])


def test_exposure_groups_cover_portfolio():
    score = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    exposure = pd.Series([1.0, 2.0, 1.0, 3.0, 1.0])
    groups = assign_exposure_groups(score, exposure, groups=3)
    assert groups.notna().all()
    assert groups.min() >= 1
    assert groups.max() <= 3
