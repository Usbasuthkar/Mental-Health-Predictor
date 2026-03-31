import sys, os
import numpy as np
import matplotlib.pyplot as plt
from constants import get_countries
import seaborn as sns
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from predict import predict_from_user_details

cm = np.load("artifacts/confusion_matrix.npy")

st.set_page_config(
    page_title="Mental Health Risk Assessment",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Mental Health Risk Assessment")
st.markdown(
    """
    Answer openly in your own words.
    Your written responses are analysed to estimate risk patterns.
    **Nothing is stored or shared.**
    """
)
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — About You
# ─────────────────────────────────────────────
st.subheader("👤 About You")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=16, max_value=100, value=28)

with col2:
    gender_text = st.text_input("Gender description")
countries = get_countries()
country_text = st.selectbox("Which country do you live in?",countries)

st.markdown("**Family Mental Health History**")

family_history_part1 = st.text_area(
    "1️⃣ Is there any mental health history in your immediate family?",
    height=70,
    key="fh_1"
)

family_history_part2 = st.text_area(
    "2️⃣ How has this affected you personally (if at all)?",
    height=70,
    key="fh_2"
)

# 🔗 Join both answers into one combined field
family_history_text = f"{family_history_part1.strip()} {family_history_part2.strip()}".strip()

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 2 — Work
# ─────────────────────────────────────────────
st.subheader("🏢 Your Workplace")

no_employees = st.selectbox(
    "Company size",
    options=["1-5", "6-25", "26-100", "100-500", "500-1000", "More Than 1000"],
    index=2,
)

self_employed_text = st.text_area(
    "Are you self-employed? Describe your work setup.",
    height=60
)


st.markdown("**Mental Health & Work**")

work_part1 = st.text_area(
    "1️⃣ Does your mental health affect your ability to work? If so, how?",
    height=70,
    key="work_1"
)

work_part2 = st.text_area(
    "2️⃣ When does it tend to show up most (deadlines, meetings, daily focus, etc.)?",
    height=70,
    key="work_2"
)

work_interfere_text = f"{work_part1.strip()} {work_part2.strip()}".strip()

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 3 — Employer Support (TEXT ONLY)
# ─────────────────────────────────────────────

st.markdown("**Employer Mental Health Benefits**")

benefits_part1 = st.text_area(
    "1️⃣ What mental health benefits does your employer provide?(eg: incentives)",
    height=70,
    key="ben_1"
)

benefits_part2 = st.text_area(
    "2️⃣ How accessible or useful are these benefits in practice?",
    height=70,
    key="ben_2"
)

benefits_text = f"{benefits_part1.strip()} {benefits_part2.strip()}".strip()
st.markdown("**Care Options at Work**")

care_part1 = st.text_area(
    "1️⃣ What mental health care options are available at your workplace?(eg: gym, therapy)",
    height=70,
    key="care_1"
)

care_part2 = st.text_area(
    "2️⃣ How clearly are these options communicated to employees?",
    height=70,
    key="care_2"
)

care_options_text = f"{care_part1.strip()} {care_part2.strip()}".strip()
st.markdown("**Wellness or Mental Health Programmes**")

wellness_part1 = st.text_area(
    "1️⃣ Does your employer run any wellness or mental health programmes?(eg: seminars, games)",
    height=70,
    key="well_1"
)

seek_help_text = st.text_area(
    "2️⃣ How effective or meaningful are these programmes for you?",
    height=70,
    key="well_2"
)

anonymity_text = st.text_area(
    "If you seek help, is your privacy/anonymity protected?",
    height=60
)

wellness_text = wellness_part1
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 4 — Personal Support
# ─────────────────────────────────────────────
st.subheader("🤝 Personal Support Network")

st.markdown("**Personal Support Network**")

support_part1 = st.text_area(
    "1️⃣ Who do you rely on for emotional support?",
    height=70,
    key="sup_1"
)

support_part2 = st.text_area(
    "2️⃣ How comfortable do you feel opening up to them?",
    height=70,
    key="sup_2"
)

support_desc = f"{support_part1.strip()} {support_part2.strip()}".strip()

leave_text = st.text_area(
    "How easy is it for you to take leave for mental health reasons?",
    height=60
)

support_score = st.slider(
    "How supported do you feel overall? (1 = isolated → 5 = very supported)",
    min_value=1,
    max_value=5,
    value=3,
)

st.markdown("---")

submitted = st.button("🔍 Assess My Mental Health Risk", type="primary", use_container_width=True)

if submitted:

    if not gender_text.strip():
        st.warning("Please enter gender description.")
        st.stop()

    user_details = {
        "age": age,
        "gender": gender_text,
        "country": country_text,
        "self_employed_text": self_employed_text,
        "family_history_text": family_history_text,
        "no_employees": no_employees,
        "work_interfere_text": work_interfere_text,
        "benefits_text": benefits_text,
        "care_options_text": care_options_text,
        "wellness_text": wellness_text,
        "seek_help_text": seek_help_text,
        "anonymity_text": anonymity_text,
        "leave_text": leave_text,
        "support_score": support_score,
    }

    with st.spinner("Analysing responses..."):
        result = predict_from_user_details(user_details)

    proba = result["pred_proba"]
    suggestions = result["suggestions"]

    if proba >= 70:
        colour, label, icon = "#e74c3c", "High Risk", "🔴"
    elif proba >= 45:
        colour, label, icon = "#f39c12", "Moderate Risk", "🟡"
    else:
        colour, label, icon = "#27ae60", "Low Risk", "🟢"

    st.subheader("📊 Your Results")

    st.markdown(
        f"""
        <div style="
            background:{colour}15;
            border-left:6px solid {colour};
            border-radius:10px;
            padding:20px;
        ">
            <h2 style="color:{colour};">
                {icon} {label} — {proba}%
            </h2>
            <p>
                Estimated probability profile associated with seeking mental health support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(int(proba))
    st.markdown("---")

    st.subheader("💡 Personalised Suggestions")
    for s in suggestions:
        st.info(s)

    st.markdown("---")
    st.markdown(
        "> ⚠️ This is not a clinical diagnosis. If struggling, consult a licensed professional."
    )
    st.subheader("📊 Model Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Risk", "Risk"],
                yticklabels=["No Risk", "Risk"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

