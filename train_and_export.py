"""
train_and_export.py
--------------------
Trains the best model on cleaned_data.csv and saves all artifacts to ./artifacts/
Run this ONCE before launching the Streamlit app.

Requirements: pip install scikit-learn xgboost joblib pandas numpy
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

os.makedirs("artifacts", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
DATA_CSV = "cleaned_data.csv"
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(
        f"'{DATA_CSV}' not found. Export your notebook DataFrame first:\n"
        "    cleaned_data.to_csv('cleaned_data.csv', index=False)"
    )

print(f"Loading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV)
print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

# ── Feature set ────────────────────────────────────────────────────────────
FEATURES = [
    "age", "gender", "family_history", "no_employees", "work_interfere",
    "has_benefits", "has_care_options", "has_wellness_program", "support_score",
]
TARGET = "treatment_encoded"

missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

df = df.dropna(subset=FEATURES + [TARGET])
print(f"  Rows after dropping nulls: {len(df)}")

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

# ── Standardise categorical strings (CSV uses mixed case) ──────────────────
CAT_COLS = ["gender", "family_history", "no_employees", "work_interfere"]
for col in CAT_COLS:
    X[col] = X[col].astype(str).str.strip().str.title()

# ── Label-encode categoricals ──────────────────────────────────────────────
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  {col} classes: {list(le.classes_)}")

# ── Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── Candidate models ───────────────────────────────────────────────────────
candidates = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "RandomForest":       RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
}
if HAS_XGB:
    candidates["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)

results = {}
for name, model in candidates.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    results[name] = {"model": model, "auc": float(auc)}
    print(f"  AUC = {auc:.4f}")

best_name  = max(results, key=lambda n: results[n]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

# ── Reference probability distributions ───────────────────────────────────
all_probas = best_model.predict_proba(X_scaled)[:, 1].tolist()

if "survey_year" in df.columns:
    years       = df["survey_year"].astype(str).values
    all_years   = sorted(set(years))
    latest_yr   = all_years[-1]
    year_probas = {}
    for yr in all_years:
        mask = years == yr
        year_probas[yr] = best_model.predict_proba(X_scaled[mask])[:, 1].tolist()
    print(f"\nSurvey years: {all_years}  |  latest = {latest_yr}")
else:
    latest_yr   = "all"
    year_probas = {"all": all_probas}

# ── Defaults ───────────────────────────────────────────────────────────────
defaults = {
    "support_score_mean": float(X["support_score"].mean()),
    "latest_year":        latest_yr,
    "best_model_name":    best_name,
    "best_auc":           round(results[best_name]["auc"], 4),
    "total_rows":         len(df),
}

# ── Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(best_model,     "artifacts/model.pkl")
joblib.dump(scaler,         "artifacts/scaler.pkl")
joblib.dump(label_encoders, "artifacts/label_encoders.pkl")
with open("artifacts/defaults.json",      "w") as f: json.dump(defaults,     f, indent=2)
with open("artifacts/ref_probas_all.json","w") as f: json.dump(all_probas,   f)
with open("artifacts/ref_probas_year.json","w") as f: json.dump(year_probas, f)

print("\nAll artifacts saved to ./artifacts/")
print(json.dumps(defaults, indent=2))