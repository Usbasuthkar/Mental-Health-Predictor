# """
# predict.py
# -----------
# Loads artifacts from ./artifacts/ and exposes predict_from_user_details().
# All three original bugs are fixed here.
# """

# import os
# import copy
# import json
# import joblib
# import numpy as np
# import random
# import pandas as pd

# ARTIFACT_DIR   = "artifacts"
# MODEL_PATH     = os.path.join(ARTIFACT_DIR, "model.pkl")
# SCALER_PATH    = os.path.join(ARTIFACT_DIR, "scaler.pkl")
# LE_PATH        = os.path.join(ARTIFACT_DIR, "label_encoders.pkl")
# DEFAULTS_PATH  = os.path.join(ARTIFACT_DIR, "defaults.json")
# REF_ALL_PATH   = os.path.join(ARTIFACT_DIR, "ref_probas_all.json")
# REF_YEAR_PATH  = os.path.join(ARTIFACT_DIR, "ref_probas_year.json")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         "Model artifact not found. Run train_and_export.py first."
#     )

# model          = joblib.load(MODEL_PATH)
# scaler         = joblib.load(SCALER_PATH)
# label_encoders = joblib.load(LE_PATH)

# with open(DEFAULTS_PATH)  as f: defaults      = json.load(f)
# with open(REF_ALL_PATH)   as f: ref_all       = json.load(f)
# with open(REF_YEAR_PATH)  as f: ref_year_dict = json.load(f)

# FEATURES = [
#     "age", "gender", "family_history", "no_employees", "work_interfere",
#     "has_benefits", "has_care_options", "has_wellness_program", "support_score",
# ]

# # Canonical title-case strings that exist in the training data
# _WORK_INTERFERE_DEFAULT = "Never"
# _FAMILY_HISTORY_DEFAULT = "No"
# _GENDER_DEFAULT         = "Other"
# _NO_EMPLOYEES_DEFAULT   = "1-5"


# def _to_bool_yes(value, yes_values=("yes",)):
#     """Map a free-text employer-question answer to 0 or 1."""
#     if value is None:
#         return 0
#     s = str(value).strip().lower()
#     return 1 if s in {v.lower() for v in yes_values} else 0


# def _title(value, fallback):
#     """Safely title-case a string, return fallback if blank/None."""
#     if not value:
#         return fallback
#     return str(value).strip().title()


# def preprocess_user_details(user_details: dict) -> np.ndarray:
#     row = {}

#     # ── Numeric ───────────────────────────────────────────────────────────
#     row["age"] = float(user_details.get("age") or 0)

#     # ── Categoricals — must be title-cased to match training LabelEncoders ─
#     row["gender"]         = _title(user_details.get("gender"),         _GENDER_DEFAULT)
#     row["family_history"] = _title(user_details.get("family_history"), _FAMILY_HISTORY_DEFAULT)
#     row["no_employees"]   = _title(user_details.get("no_employees"),   _NO_EMPLOYEES_DEFAULT)

#     # FIX 1 & 2: was calling predict_from_user_details(sample) on an undefined
#     # variable inside this function; also never assigned to row['work_interfere']
#     wi = user_details.get("work_interfere")
#     row["work_interfere"] = _title(wi, _WORK_INTERFERE_DEFAULT)

#     # ── Boolean employer flags ─────────────────────────────────────────────
#     row["has_benefits"]         = _to_bool_yes(user_details.get("benefits"),         yes_values=("Yes", "yes"))
#     row["has_care_options"]     = _to_bool_yes(user_details.get("care_options"),     yes_values=("Yes", "yes"))
#     row["has_wellness_program"] = _to_bool_yes(user_details.get("wellness_program"), yes_values=("Yes", "yes"))

#     # ── Support score ──────────────────────────────────────────────────────
#     support = user_details.get("support_score")
#     row["support_score"] = (
#         defaults.get("support_score_mean", 0.0)
#         if (support is None or str(support).strip() == "")
#         else float(support)
#     )

#     # ── Build DataFrame ────────────────────────────────────────────────────
#     df = pd.DataFrame([row], columns=FEATURES)

#     # ── Encode categoricals ────────────────────────────────────────────────
#     for col, le in label_encoders.items():
#         val = str(df.at[0, col])
#         if val not in le.classes_:
#             # FIX 3: joblib has no loads/dumps API — use copy.deepcopy
#             le_ext = copy.deepcopy(le)
#             le_ext.classes_ = np.array(list(le.classes_) + [val], dtype=object)
#             df[col] = le_ext.transform(df[col].astype(str))
#         else:
#             df[col] = le.transform(df[col].astype(str))

#     return scaler.transform(df)


# def _percentile_vs(user_proba: float, population: list) -> float:
#     """Percentage of the population that has LOWER risk than this user."""
#     arr = np.array(population)
#     return round(float(np.mean(arr < user_proba) * 100), 1)
# HIGH_RISK_MESSAGES = [
#     "🔴 Strong risk signal detected. Profiles similar to yours often benefit from structured professional support.",
#     "🔴 Elevated likelihood pattern. Early mental health support could make a meaningful difference.",
#     "🔴 Your responses align with higher treatment-seeking groups in the dataset.",
# ]

# MODERATE_RISK_MESSAGES = [
#     "🟡 Moderate risk pattern. A proactive check-in with a counsellor could be beneficial.",
#     "🟡 Some indicators suggest increased vulnerability. Preventative action may help.",
#     "🟡 You show a mix of protective and risk factors. Monitoring your wellbeing is wise.",
# ]

# LOW_RISK_MESSAGES = [
#     "🟢 Lower overall risk profile. Continue maintaining healthy coping habits.",
#     "🟢 Your responses align more with lower treatment-seeking groups.",
#     "🟢 Current indicators suggest relative stability — keep reinforcing protective routines.",
# ]

# WORK_MESSAGES = [
#     "💼 Mental health appears to impact your work. Flexible scheduling or workload adjustments may help.",
#     "💼 Work interference is an important signal. Structured coping strategies could reduce strain.",
#     "💼 Since work is affected, discussing realistic expectations with your manager may help.",
# ]

# NO_BENEFITS_MESSAGES = [
#     "📋 Limited employer benefits detected. Consider public or low-cost therapy options.",
#     "📋 Without employer coverage, community clinics or telehealth platforms may help.",
#     "📋 Explore independent insurance or digital therapy services.",
# ]

# NO_WELLNESS_MESSAGES = [
#     "🧘 Building structured wellness habits can compensate for limited workplace programmes.",
#     "🧘 Regular exercise, mindfulness, and sleep hygiene can reduce long-term risk.",
#     "🧘 Self-guided CBT or meditation apps may help strengthen resilience.",
# ]

# FAMILY_HISTORY_MESSAGES = [
#     "🧬 Family history increases vulnerability — proactive check-ins are powerful.",
#     "🧬 Genetic predisposition does not guarantee outcomes, but awareness is protective.",
#     "🧬 Periodic mental health screenings may be beneficial.",
# ]

# LOW_SUPPORT_MESSAGES = [
#     "🤝 Expanding your support network could significantly reduce long-term risk.",
#     "🤝 Even one trusted connection can buffer stress effects.",
#     "🤝 Peer groups or structured communities may help.",
# ]

# MID_RANGE_PROBABILITY_MESSAGES = [
#     "📊 Small lifestyle improvements now could meaningfully reduce long-term probability.",
#     "📊 Your risk sits near the mid-range — prevention strategies could shift it downward.",
# ]

# def _add_risk_message(p, suggestions):
#     if p >= 70:
#         suggestions.append(random.choice(HIGH_RISK_MESSAGES))
#     elif p >= 45:
#         suggestions.append(random.choice(MODERATE_RISK_MESSAGES))
#     else:
#         suggestions.append(random.choice(LOW_RISK_MESSAGES))


# def _add_work_message(user_details, suggestions):
#     wi = (user_details.get("work_interfere") or "").lower()
#     if wi in ("often", "sometimes"):
#         suggestions.append(random.choice(WORK_MESSAGES))


# def _add_employer_messages(user_details, suggestions):
#     has_benefits = _to_bool_yes(user_details.get("benefits"), yes_values=("Yes", "yes"))
#     has_wellness = _to_bool_yes(user_details.get("wellness_program"), yes_values=("Yes", "yes"))

#     if not has_benefits:
#         suggestions.append(random.choice(NO_BENEFITS_MESSAGES))

#     if not has_wellness:
#         suggestions.append(random.choice(NO_WELLNESS_MESSAGES))

#     # 🔥 Combined logic (multi-factor aware)
#     if not has_benefits and not has_wellness:
#         suggestions.append(
#             "🏢 With limited workplace mental health infrastructure, external structured support may be especially important."
#         )


# def _add_family_history(user_details, suggestions):
#     if (user_details.get("family_history") or "").strip().lower() == "yes":
#         suggestions.append(random.choice(FAMILY_HISTORY_MESSAGES))


# def _add_support_message(user_details, suggestions):
#     try:
#         score = float(user_details.get("support_score") or 0)
#         if score <= 2:
#             suggestions.append(random.choice(LOW_SUPPORT_MESSAGES))
#     except (ValueError, TypeError):
#         pass


# def _add_mid_range_message(p, suggestions):
#     if 40 <= p <= 60:
#         suggestions.append(random.choice(MID_RANGE_PROBABILITY_MESSAGES))

# def generate_suggestions(user_details: dict, pred_proba_pct: float) -> list:
#     suggestions = []
#     p = pred_proba_pct

#     # Build modularly
#     _add_risk_message(p, suggestions)
#     _add_work_message(user_details, suggestions)
#     _add_employer_messages(user_details, suggestions)
#     _add_family_history(user_details, suggestions)
#     _add_support_message(user_details, suggestions)
#     _add_mid_range_message(p, suggestions)

#     # 🎲 Controlled randomization
#     random.shuffle(suggestions)

#     # Limit to avoid overwhelming the user
#     return suggestions[:5]


# def predict_from_user_details(user_details: dict) -> dict:
#     X_scaled = preprocess_user_details(user_details)

#     proba_raw = float(model.predict_proba(X_scaled)[:, 1][0])   # 0.0 – 1.0
#     label     = int(model.predict(X_scaled)[0])
#     proba_pct = round(proba_raw * 100, 1)                        # 0 – 100

#     latest_yr   = defaults.get("latest_year", "all")
#     year_probas = ref_year_dict.get(str(latest_yr), ref_all)

#     pct_all  = _percentile_vs(proba_raw, ref_all)
#     pct_year = _percentile_vs(proba_raw, year_probas)

#     return {
#         "pred_label":      label,
#         "pred_proba":      proba_pct,        # risk % (0–100)
#         "percentile_all":  pct_all,           # higher risk than X% of all-time respondents
#         "percentile_year": pct_year,          # higher risk than X% of latest-year respondents
#         "latest_year":     latest_yr,
#         "suggestions":     generate_suggestions(user_details, proba_pct),
#         "total_reference": defaults.get("total_rows", "?"),
#         "model_name":      defaults.get("best_model_name", "Unknown"),
#     }


# # ── Quick smoke-test ───────────────────────────────────────────────────────
# if __name__ == "__main__":
#     sample = {
#         "age":              30,
#         "gender":           "Male",
#         "family_history":   "Yes",
#         "no_employees":     "26-100",
#         "work_interfere":   "Sometimes",
#         "benefits":         "Yes",
#         "care_options":     "Yes",
#         "wellness_program": "No",
#         "support_score":    3,
#     }
#     r = predict_from_user_details(sample)
#     print(f"\nRisk probability : {r['pred_proba']}%")
#     print(f"vs ALL years     : higher than {r['percentile_all']}% of all respondents")
#     print(f"vs {r['latest_year']}        : higher than {r['percentile_year']}% of that year's respondents")
#     print(f"Model used       : {r['model_name']}")
#     print("\nSuggestions:")
#     for s in r["suggestions"]:
#         print(" •", s)

"""
predict.py
Text-inference + 20+ rule engine version
"""

import os
import json
import joblib
import numpy as np
import random
import pandas as pd
import copy

ARTIFACT_DIR = "artifacts"
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))

with open(os.path.join(ARTIFACT_DIR, "defaults.json")) as f:
    defaults = json.load(f)

FEATURES = [
    "age","gender","family_history","no_employees",
    "work_interfere","has_benefits","has_care_options",
    "has_wellness_program","support_score"
]

# ─────────────────────────────────────────────
# TEXT INFERENCE
# ─────────────────────────────────────────────

def infer_binary(text):
    if not text:
        return 0
    s = text.lower()
    positive = ["yes","have","available","provided","offer","eap","counsellor"]
    negative = ["no","none","not","nothing","never","don't","unclear"]
    pos = sum(w in s for w in positive)
    neg = sum(w in s for w in negative)
    return 1 if pos > neg else 0

def infer_work_interference(text):
    if not text:
        return "Never"
    s = text.lower()
    if any(w in s for w in ["often","frequent","constant"]):
        return "Often"
    if any(w in s for w in ["sometimes","occasionally","periodically"]):
        return "Sometimes"
    if any(w in s for w in ["rarely"]):
        return "Rarely"
    return "Never"

def infer_family(text):
    if not text:
        return "No"
    s = text.lower()
    keywords = ["depression","anxiety","bipolar","ptsd","suicide","schizophrenia"]
    return "Yes" if any(k in s for k in keywords) else "No"

# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def preprocess_user_details(user):
    row = {}

    row["age"] = float(user.get("age",0))
    row["gender"] = str(user.get("gender","Other")).title()
    row["family_history"] = infer_family(user.get("family_history_text"))
    row["no_employees"] = str(user.get("no_employees","1-5")).title()
    row["work_interfere"] = infer_work_interference(user.get("work_interfere_text"))

    row["has_benefits"] = infer_binary(user.get("benefits_text"))
    row["has_care_options"] = infer_binary(user.get("care_options_text"))
    row["has_wellness_program"] = infer_binary(user.get("wellness_text"))
    row["support_score"] = float(user.get("support_score",3))

    df = pd.DataFrame([row], columns=FEATURES)

    for col, le in label_encoders.items():
        val = str(df.at[0,col])
        if val not in le.classes_:
            le_ext = copy.deepcopy(le)
            le_ext.classes_ = np.array(list(le.classes_) + [val], dtype=object)
            df[col] = le_ext.transform(df[col].astype(str))
        else:
            df[col] = le.transform(df[col].astype(str))

    return scaler.transform(df)

# ─────────────────────────────────────────────
# ADVANCED SUGGESTION ENGINE
# ─────────────────────────────────────────────

def generate_suggestions(user, p):

    suggestions = []
    text_blob = " ".join([
        user.get("family_history_text",""),
        user.get("work_interfere_text",""),
        user.get("benefits_text",""),
        user.get("care_options_text",""),
        user.get("wellness_text",""),
    ]).lower()

    high = p >= 70
    moderate = 45 <= p < 70
    low = p < 45

    work_flag = infer_work_interference(user.get("work_interfere_text")) in ["Often","Sometimes"]
    no_benefits = infer_binary(user.get("benefits_text")) == 0
    no_wellness = infer_binary(user.get("wellness_text")) == 0
    family_flag = infer_family(user.get("family_history_text")) == "Yes"
    low_support = float(user.get("support_score",3)) <= 2

    if high and low_support:
        suggestions.append("🔴 High risk combined with low support suggests structured professional help may be critical.")

    if high and work_flag:
        suggestions.append("🔴 Elevated risk with work strain detected — workplace adjustments could reduce escalation.")

    if moderate and family_flag:
        suggestions.append("🧬 Family vulnerability + moderate profile — preventative therapy could reduce long-term risk.")

    if no_benefits and low_support:
        suggestions.append("⚠️ Limited workplace and personal support — external support options are especially important.")

    if high and no_benefits and no_wellness:
        suggestions.append("🚨 Multi-factor vulnerability detected — proactive intervention strongly recommended.")

    burnout_words = ["burnout","exhausted","overwhelmed","drained","hopeless"]
    if any(w in text_blob for w in burnout_words):
        suggestions.append("🔥 Language indicates burnout signals — recovery planning and rest cycles may help.")

    if low:
        suggestions.append("🟢 Maintain protective habits — regular sleep, movement, and social connection reinforce resilience.")

    if moderate:
        suggestions.append("🟡 Mid-range risk suggests preventative action now could shift long-term trajectory.")

    if high:
        suggestions.append("🔴 Profile resembles high treatment-seeking groups in dataset.")

    if family_flag:
        suggestions.append("🧬 Family history awareness allows early monitoring and protective behaviour.")

    if work_flag:
        suggestions.append("💼 Work interference signals strain — structured coping tools may reduce daily stress.")

    if low_support:
        suggestions.append("🤝 Expanding support network could significantly buffer long-term vulnerability.")

    random.shuffle(suggestions)
    return suggestions[:8]  # return up to 8 strong suggestions

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict_from_user_details(user):

    X = preprocess_user_details(user)
    proba_raw = float(model.predict_proba(X)[:,1][0])
    proba_pct = round(proba_raw*100,1)

    return {
        "pred_proba": proba_pct,
        "suggestions": generate_suggestions(user, proba_pct)
    }