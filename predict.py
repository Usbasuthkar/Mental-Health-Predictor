import os
import json
import joblib
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import random
import pandas as pd
import copy
from dotenv import load_dotenv

load_dotenv()
ARTIFACT_DIR = "artifacts"
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))

with open(os.path.join(ARTIFACT_DIR, "defaults.json")) as f:
    defaults = json.load(f)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

parser = JsonOutputParser()
pre_prompt = ChatPromptTemplate.from_messages([
    ("system", """Return ONLY valid JSON.

Convert input JSON into:
{
  "family_history": "Yes or No",
  "work_interfere": "Yes or No",
  "benefits": "Yes or No",
  "care_options": "Yes or No",
  "wellness": "Yes or No",
  "self_employed": "Yes or No",
  "tech_company": "Yes or No",
  "seek_help": "Yes or No",
  "anonymity": "Yes or No",
  "leave": "Very easy / Somewhat easy / Difficult / Very difficult / Don't know"
}
"""),
    ("human", "JSON:{user_details}"),
])

post_prompt = ChatPromptTemplate.from_messages([
    ("system","""
     You are a psychiatrist helping an employee. 
     Based on the given JSON data and a 
     prediction of high risk for mental health issues, 
     provide a practical suggestion to improve mental well-being. 
     MAX 50 words.
      """),
    ("human","JSON:{user_details}, prediction:{value}"),
])

pre_processing_chain  = pre_prompt | llm | parser
post_processing_chain = post_prompt | llm

FEATURES = [
    "age",
    "gender",
    "country",
    "self_employed",
    "family_history",
    "no_employees",
    "work_interfere",
    "has_benefits",
    "has_care_options",
    "has_wellness_program",
    "seek_help",
    "anonymity",
    "leave",
    "support_score"
]
# ─────────────────────────────────────────────
# TEXT INFERENCE
# ─────────────────────────────────────────────
def infer(user):
    result = pre_processing_chain.invoke({"user_details": user})
    return result
# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
def yes_no_to_int(val):
    return 1 if str(val).strip().lower() == "yes" else 0

def preprocess_user_details(user):
    row = {}
    row["age"] = float(user.get("age",0))
    row["gender"] = str(user.get("gender","Other")).title()
    row["no_employees"] = str(user.get("no_employees","1-5")).title()
    row["support_score"] = float(user.get("support_score",3))
    row["country"] = str(user.get("country", "Other")).title()

    exclude_keys = {"age", "gender", "no_employees", "support_score"}
    infer_input = {k: v for k, v in user.items() if k not in exclude_keys}
    result = infer(infer_input)
    row["self_employed"] = yes_no_to_int(result.get("self_employed"))

    row["seek_help"] = yes_no_to_int(result.get("seek_help"))
    row["anonymity"] = yes_no_to_int(result.get("anonymity"))

    row["leave"] = str(result.get("leave", "Don't know")).title()
    row["family_history"] = yes_no_to_int(result.get("family_history"))
    row["work_interfere"] = yes_no_to_int(result.get("work_interfere"))

    row["has_benefits"] = yes_no_to_int(result.get("benefits"))
    row["has_care_options"] = yes_no_to_int(result.get("care_options"))
    row["has_wellness_program"] = yes_no_to_int(result.get("wellness"))

    df = pd.DataFrame([row])

    # ensure all features exist
    for col in FEATURES:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)

    # enforce order
    df = df[FEATURES]
    print(row)

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
    result = post_processing_chain.invoke({"user_details":user,"value":p})
    suggestions.append(result.content)
    return suggestions

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