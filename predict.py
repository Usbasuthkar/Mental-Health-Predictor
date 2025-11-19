import os
import json
import joblib
import numpy as np
import pandas as pd


ARTIFACT_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pkl')
SCALER_PATH = os.path.join(ARTIFACT_DIR, 'scaler.pkl')
LE_PATH = os.path.join(ARTIFACT_DIR, 'label_encoders.pkl')
DEFAULTS_PATH = os.path.join(ARTIFACT_DIR, 'defaults.json')


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError('Model artifact not found. Run train_and_export.py first to generate artifacts in ./artifacts')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(LE_PATH)
with open(DEFAULTS_PATH, 'r') as f:
    defaults = json.load(f)


FEATURES = ['age', 'gender', 'family_history', 'no_employees', 'work_interfere',
'has_benefits', 'has_care_options', 'has_wellness_program', 'support_score']

def _to_bool_yes(value, yes_values=('yes',)):
    if value is None:
        return 0
    s = str(value).strip().lower()
    return 1 if any(s == v for v in [v.lower() for v in yes_values]) else 0


def preprocess_user_details(user_details: dict):
    row = {}
    row['age'] = float(user_details.get('age') or 0)
    row['gender'] = user_details.get('gender') or 'Other'
    row['family_history'] = user_details.get('family_history') or 'No'
    row['no_employees'] = user_details.get('no_employees') or '1-5'
    wi = user_details.get('work_interfere')
    if not wi:
        print(predict_from_user_details(sample))
    else:
        row['work_interfere'] = wi

    row['has_benefits'] = _to_bool_yes(user_details.get('benefits'), yes_values=('Yes','yes'))  
    row['has_care_options'] = _to_bool_yes(user_details.get('care_options'), yes_values=('Yes','yes'))
    row['has_wellness_program'] = _to_bool_yes(user_details.get('wellness_program'), yes_values=('Yes','yes'))


    support = user_details.get('support_score')
    if support is None or str(support).strip() == '':
        row['support_score'] = defaults.get('support_score_mean', 0.0)
    else:
        row['support_score'] = float(support)


    df = pd.DataFrame([row], columns=FEATURES)

    for col, le in label_encoders.items():
        val = str(df.at[0, col])
        if val not in le.classes_:
            classes = list(le.classes_) + [val]
            le_extended = joblib.loads(joblib.dumps(le))
            le_extended.classes_ = np.array(classes, dtype=object)
            df[col] = le_extended.transform(df[col].astype(str))
        else:
            df[col] = le.transform(df[col].astype(str))

    X_scaled = scaler.transform(df)
    return X_scaled

def predict_from_user_details(user_details: dict):
    X_scaled = preprocess_user_details(user_details)
    proba = model.predict_proba(X_scaled)[:, 1][0]
    label = int(model.predict(X_scaled)[0])
    return {'pred_label': label, 'pred_proba': float(proba)}

if __name__ == '__main__':
    sample = {
        'age': 30,
        'gender': 'Male',
        'family_history': 'Yes',
        'no_employees': '26-100',
        'work_interfere': 'Sometimes',
        'benefits': 'Yes',
        'care_options': 'Yes',
        'wellness_program': 'No',
        }
    print(predict_from_user_details(sample))