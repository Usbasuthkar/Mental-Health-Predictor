import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

os.makedirs('artifacts', exist_ok=True)

DATA_CSV = 'cleaned_data.csv'
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError("Please export your cleaned_data DataFrame from the notebook to 'cleaned_data.csv' before running this script.\nIn your notebook: cleaned_data.to_csv('cleaned_data.csv', index=False)")

print('Loading', DATA_CSV)
df = pd.read_csv(DATA_CSV)

# FEATURES used during training (must match the notebook)
features = ['age', 'gender', 'family_history', 'no_employees', 'work_interfere',
            'has_benefits', 'has_care_options', 'has_wellness_program', 'support_score']

# target column used in your notebook
TARGET = 'treatment_encoded'

# Ensure columns exist
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in cleaned_data.csv: {missing}")

X = df[features].copy()
y = df[TARGET].copy()

# Fit LabelEncoders for categorical columns and save mapping
cat_cols = ['gender', 'family_history', 'no_employees', 'work_interfere']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Persist default (mean) for support_score and any other defaults
defaults = {
    'support_score_mean': float(X['support_score'].mean()),
}

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Candidate models
models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    print('Training', name)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'roc_auc': float(auc)}
    print(f'  AUC = {auc:.4f}')

# Pick best model by roc_auc
best_name = max(results.keys(), key=lambda n: results[n]['roc_auc'])
best_model = results[best_name]['model']
print('\nBest model:', best_name, 'AUC=', results[best_name]['roc_auc'])

# Save artifacts
joblib.dump(best_model, 'artifacts/model.pkl')
joblib.dump(scaler, 'artifacts/scaler.pkl')
joblib.dump(label_encoders, 'artifacts/label_encoders.pkl')
with open('artifacts/defaults.json', 'w') as f:
    json.dump(defaults, f)

print('Saved artifacts to ./artifacts/')


