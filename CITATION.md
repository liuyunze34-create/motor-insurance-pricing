# Data and references

This portfolio project uses the French motor third-party liability datasets available through OpenML:

- `freMTPL2freq` — policy exposure, claim counts and rating variables.
- `freMTPL2sev` — individual claim amounts.

The frequency/severity structure is based on the widely used French MTPL pricing case study. The project extends that setup with explicit source reconciliation, repeated model-family validation, a nonlinear challenger, prediction-based pricing relativities, calibration diagnostics and a dashboard.

For reproducibility, `download_data.py` retrieves the source tables directly from OpenML using dataset IDs 41214 and 41215.
