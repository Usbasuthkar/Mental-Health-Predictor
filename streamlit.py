
# """
# streamlit_app.py
# -----------------
# Mental Health Risk Assessment — clean Streamlit form.
# Run with: streamlit run streamlit_app.py

# All Yes/No questions are split into two parts:
#   1. A dropdown for the core factual answer
#   2. A free-text box for open-ended context

# Results show:
#   • Risk % score (colour-coded)
#   • Percentile vs this year's survey respondents
#   • Percentile vs all-time respondents
#   • Personalised suggestions based on every response
# """

# import sys, os
# sys.path.insert(0, os.path.dirname(__file__))

# import streamlit as st
# from predict import predict_from_user_details

# # ── Page config ────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Mental Health Risk Assessment",
#     page_icon="🧠",
#     layout="centered",
# )

# # ── Header ─────────────────────────────────────────────────────────────────
# st.title("🧠 Mental Health Risk Assessment")
# st.markdown(
#     """
#     Answer each question honestly — there are no right or wrong responses.
#     Your answers are used only to generate a personalised risk estimate and suggestions.
#     **Nothing is stored or shared.**
#     """,
#     unsafe_allow_html=False,
# )
# st.markdown("---")


# # ══════════════════════════════════════════════════════════════════════════
# # SECTION 1 — About You
# # ══════════════════════════════════════════════════════════════════════════
# st.subheader("👤 About You")

# col1, col2 = st.columns(2)
# with col1:
#     age = st.number_input(
#         "How old are you?",
#         min_value=16, max_value=100, value=28, step=1,
#     )
# with col2:
#     gender_text = st.text_input(
#         "How do you describe your gender?",
#         placeholder="e.g. Male, Female, Non-binary, Prefer not to say …",
#     )

# # ── Family history — split into 2 ─────────────────────────────────────────
# st.markdown("**Family mental health history**")

# fh_aware = st.selectbox(
#     "Are you aware of any mental health conditions in your immediate family?",
#     options=["No", "Yes", "Unsure / prefer not to say"],
#     index=0,
# )

# fh_context = ""
# if fh_aware == "Yes":
#     fh_context = st.text_area(
#         "Briefly, how has that family history affected your own outlook on mental health?",
#         placeholder=(
#             "e.g. It made me more proactive about seeking help, "
#             "it causes me some anxiety, I try not to think about it …"
#         ),
#         height=80,
#         key="fh_context",
#     )

# family_history = "Yes" if fh_aware == "Yes" else "No"

# st.markdown("---")


# # ══════════════════════════════════════════════════════════════════════════
# # SECTION 2 — Your Workplace
# # ══════════════════════════════════════════════════════════════════════════
# st.subheader("🏢 Your Workplace")

# no_employees = st.selectbox(
#     "Roughly how many people work at your company?",
#     options=["1-5", "6-25", "26-100", "100-500", "500-1000", "More Than 1000"],
#     index=2,
# )

# # ── Work interference — split into 2 ──────────────────────────────────────
# st.markdown("**Work & mental health**")

# wi_frequency = st.selectbox(
#     "Does your mental health currently interfere with your ability to do your work?",
#     options=["Never", "Rarely", "Sometimes", "Often"],
#     index=0,
# )

# wi_context = st.text_area(
#     "Describe briefly how it affects (or doesn't affect) your day-to-day work:",
#     placeholder=(
#         "e.g. Hard to focus during stressful deadlines, "
#         "occasional anxiety before big meetings, generally fine …"
#     ),
#     height=80,
#     key="wi_context",
# )

# work_interfere = wi_frequency  # already title-cased

# st.markdown("---")


# # ══════════════════════════════════════════════════════════════════════════
# # SECTION 3 — Employer Mental Health Support
# # (all originally Yes/No — each split into 2 questions)
# # ══════════════════════════════════════════════════════════════════════════
# st.subheader("🏥 Employer Support")

# # ── Benefits ───────────────────────────────────────────────────────────────
# ben_yn = st.selectbox(
#     "Does your employer provide any mental health benefits as part of health coverage?",
#     options=["Not sure", "Yes", "No"],
#     index=0,
#     key="ben_yn",
# )
# ben_context = st.text_input(
#     "What do you know about those benefits — or why they seem to be absent?",
#     placeholder=(
#         "e.g. We have an EAP, therapy is partially covered, "
#         "HR has never mentioned it …"
#     ),
#     key="ben_ctx",
# )
# benefits = ben_yn  # "Yes" / "No" / "Not sure"

# st.markdown("")

# # ── Care options ───────────────────────────────────────────────────────────
# care_yn = st.selectbox(
#     "Are you aware of the mental health care options your employer provides?",
#     options=["Not sure", "Yes", "No"],
#     index=0,
#     key="care_yn",
# )
# care_context = st.text_input(
#     "What options are you aware of — or what makes them unclear?",
#     placeholder=(
#         "e.g. In-house counsellor, referral list available, "
#         "nothing has ever been communicated …"
#     ),
#     key="care_ctx",
# )
# care_options = care_yn

# st.markdown("")

# # ── Wellness programme ──────────────────────────────────────────────────────
# well_yn = st.selectbox(
#     "Has your employer ever formally run a mental health or employee wellness programme?",
#     options=["Not sure", "Yes", "No"],
#     index=0,
#     key="well_yn",
# )
# well_context = st.text_input(
#     "Tell us a bit more about that programme — or why one doesn't seem to exist:",
#     placeholder=(
#         "e.g. Annual wellness day, trained mental health first-aiders, "
#         "yoga sessions, nothing at all …"
#     ),
#     key="well_ctx",
# )
# wellness_program = well_yn

# st.markdown("---")


# # ══════════════════════════════════════════════════════════════════════════
# # SECTION 4 — Personal Support Network
# # ══════════════════════════════════════════════════════════════════════════
# st.subheader("🤝 Personal Support Network")

# support_desc = st.text_area(
#     "Describe your personal support network (friends, family, therapists, community groups, etc.):",
#     placeholder=(
#         "e.g. I have close friends I can talk to openly, "
#         "I see a therapist monthly, my family is supportive but we don't discuss feelings …"
#     ),
#     height=90,
# )

# support_score = st.slider(
#     "How supported do you feel overall right now?  (1 = very isolated → 5 = very well supported)",
#     min_value=1, max_value=5, value=3,
# )

# st.markdown("---")


# # ══════════════════════════════════════════════════════════════════════════
# # SUBMIT
# # ══════════════════════════════════════════════════════════════════════════
# submitted = st.button("🔍 Assess My Mental Health Risk", type="primary", use_container_width=True)

# if submitted:
#     # ── Basic validation ────────────────────────────────────────────────────
#     if not gender_text.strip():
#         st.warning("Please describe your gender before submitting (even 'prefer not to say' is fine).")
#         st.stop()

#     user_details = {
#         "age":              age,
#         "gender":           gender_text.strip() or "Other",
#         "family_history":   family_history,
#         "no_employees":     no_employees,
#         "work_interfere":   work_interfere,
#         "benefits":         benefits,
#         "care_options":     care_options,
#         "wellness_program": wellness_program,
#         "support_score":    support_score,
#     }

#     with st.spinner("Analysing your responses …"):
#         try:
#             result = predict_from_user_details(user_details)
#         except FileNotFoundError as e:
#             st.error(str(e))
#             st.stop()

#     proba      = result["pred_proba"]           # 0–100
#     pct_all    = result["percentile_all"]        # higher than X% of all respondents
#     pct_year   = result["percentile_year"]
#     latest_yr  = result["latest_year"]
#     suggestions= result["suggestions"]
#     n_ref      = result["total_reference"]
#     model_nm   = result["model_name"]

#     # ── Risk band ───────────────────────────────────────────────────────────
#     if proba >= 70:
#         colour, risk_label, risk_icon = "#e74c3c", "High",     "🔴"
#     elif proba >= 45:
#         colour, risk_label, risk_icon = "#f39c12", "Moderate", "🟡"
#     else:
#         colour, risk_label, risk_icon = "#27ae60", "Low",      "🟢"

#     st.markdown("---")
#     st.subheader("📊 Your Results")

#     # ── Big risk banner ─────────────────────────────────────────────────────
#     st.markdown(
#         f"""
#         <div style="
#             background:{colour}15;
#             border-left:6px solid {colour};
#             border-radius:10px;
#             padding:20px 24px;
#             margin-bottom:20px;
#         ">
#             <h2 style="color:{colour}; margin:0 0 6px 0;">
#                 {risk_icon} {risk_label} Risk &nbsp;—&nbsp; {proba}%
#             </h2>
#             <p style="margin:0; color:#444; font-size:1rem;">
#                 Based on your responses, our model estimates a
#                 <strong>{proba}% probability</strong> that your profile
#                 is associated with seeking mental health treatment or support.
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # ── Progress bar ────────────────────────────────────────────────────────
#     st.progress(int(proba))
#     st.caption(f"Model: {model_nm}  |  Reference dataset: {n_ref:,} respondents")

#     st.markdown("---")

#     # ── Percentile comparison cards ─────────────────────────────────────────
#     st.subheader("📈 How You Compare to Others")
#     st.markdown(
#         "These figures show what percentage of survey respondents have a **lower** "
#         "risk score than yours — a higher number means your risk is elevated relative "
#         "to more of the population."
#     )

#     c1, c2 = st.columns(2)

#     with c1:
#         st.markdown(
#             f"""
#             <div style="
#                 background:#f0f4ff;
#                 border:1px solid #c5d0f5;
#                 border-radius:10px;
#                 padding:18px 20px;
#                 text-align:center;
#             ">
#                 <div style="font-size:0.85rem; color:#666; margin-bottom:4px;">
#                     📅 vs <strong>{latest_yr}</strong> respondents
#                 </div>
#                 <div style="font-size:2.2rem; font-weight:700; color:#2c3e9e;">
#                     {pct_year}%
#                 </div>
#                 <div style="font-size:0.8rem; color:#555; margin-top:4px;">
#                     of {latest_yr} respondents have a lower risk score
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     with c2:
#         st.markdown(
#             f"""
#             <div style="
#                 background:#f0f4ff;
#                 border:1px solid #c5d0f5;
#                 border-radius:10px;
#                 padding:18px 20px;
#                 text-align:center;
#             ">
#                 <div style="font-size:0.85rem; color:#666; margin-bottom:4px;">
#                     📆 vs <strong>all years</strong> combined
#                 </div>
#                 <div style="font-size:2.2rem; font-weight:700; color:#2c3e9e;">
#                     {pct_all}%
#                 </div>
#                 <div style="font-size:0.8rem; color:#555; margin-top:4px;">
#                     of all-time respondents have a lower risk score
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     st.caption(
#         "Comparisons are based on the OSMI Mental Health in Tech survey dataset. "
#         "This is a statistical tool, not a clinical assessment."
#     )

#     st.markdown("---")

#     # ── Suggestions ─────────────────────────────────────────────────────────
#     st.subheader("💡 Personalised Suggestions")
#     for tip in suggestions:
#         st.info(tip)

#     # ── Echo open-text context ───────────────────────────────────────────────
#     open_responses = [
#         ("Work experience", wi_context),
#         ("Support network", support_desc),
#         ("Family history context", fh_context),
#         ("Benefits context", ben_context),
#         ("Care options context", care_context),
#         ("Wellness programme context", well_context),
#     ]
#     open_responses = [(label, txt) for label, txt in open_responses if txt and txt.strip()]

#     if open_responses:
#         with st.expander("📝 Your open-text responses (for reference)"):
#             for label, txt in open_responses:
#                 st.markdown(f"**{label}:** {txt}")

#     st.markdown("---")
#     st.markdown(
#         "> ⚠️ **Disclaimer:** This assessment is not a clinical diagnosis. "
#         "The model is trained on self-reported survey data and is intended for "
#         "awareness purposes only. If you are struggling, please reach out to a "
#         "qualified mental health professional or a crisis line in your country."
#     )

"""
streamlit_app.py
-----------------
Mental Health Risk Assessment — TEXT-INFERENCE VERSION
All Yes/No questions removed.
Inference is performed from free-text answers.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from predict import predict_from_user_details

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
    "1️⃣ What mental health benefits does your employer provide?",
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
    "1️⃣ What mental health care options are available at your workplace?",
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
    "1️⃣ Does your employer run any wellness or mental health programmes?",
    height=70,
    key="well_1"
)

wellness_part2 = st.text_area(
    "2️⃣ How effective or meaningful are these programmes for you?",
    height=70,
    key="well_2"
)

wellness_text = f"{wellness_part1.strip()} {wellness_part2.strip()}".strip()
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
        "family_history_text": family_history_text,
        "no_employees": no_employees,
        "work_interfere_text": work_interfere_text,
        "benefits_text": benefits_text,
        "care_options_text": care_options_text,
        "wellness_text": wellness_text,
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

