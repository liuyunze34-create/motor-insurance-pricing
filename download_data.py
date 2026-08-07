from pathlib import Path
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

freq = fetch_openml(data_id=41214, as_frame=True).data.copy()
sev = fetch_openml(data_id=41215, as_frame=True).data.copy()

freq["IDpol"] = freq["IDpol"].astype(int)
sev["IDpol"] = sev["IDpol"].astype(int)

freq.to_csv(DATA_DIR / "freMTPL2freq.csv", index=False)
sev.to_csv(DATA_DIR / "freMTPL2sev.csv", index=False)

print(f"Saved {len(freq):,} policy rows and {len(sev):,} severity records to {DATA_DIR}")
