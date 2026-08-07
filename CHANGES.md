# Changes in v2

- The final test set is no longer used to choose the pricing model.
- Model-family selection now uses three repeated validation splits.
- Added source reconciliation for unmatched severity records and claim-count mismatches.
- Added explicit modeling-adjustment and large-loss concentration tables.
- Added a histogram gradient-boosting frequency/severity challenger.
- Added a Tweedie power sensitivity table.
- Risk deciles are exposure-weighted.
- Added calibration tables by driver age, bonus-malus and area.
- Replaced coefficient-style relativities with prediction-based relativities around a reference policy.
- The dashboard now separates the raw and modeling portfolios.
- The policy example shows portfolio relativity, percentile and risk decile and restricts area/region combinations to observed pairs.
- Added data download script, automated tests, GitHub Actions configuration, pinned core package versions and a license.
- Notebook, dashboard and documentation headings were simplified and stale v1 files were removed.
