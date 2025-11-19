import streamlit as st
import countries
from predict import predict_from_user_details

st.set_page_config(page_title='Mental Health Prediction', layout='centered')
st.title('Mental Health — Prediction')

user_details = {}
user_details['age'] = st.number_input("How old are you?", min_value=0, max_value=120, step=1)
user_details['gender'] = st.selectbox("Your gender??", ["", 'Male', 'Female', 'Other'])
user_details['country'] = st.selectbox("In which country do you live??", countries.get_countries())
user_details['self_employed'] = st.selectbox("Are you self-employed?", ["", "Yes", "No"])
user_details['family_history'] = st.selectbox("Do you have a family history of mental illness?", ["", "Yes", "No"])
user_details['work_interfere'] = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?", ["", "Often", "Sometimes", "Rarely", "Never", "Unknown"])
user_details['no_employees'] = st.selectbox("How many employees does your company or organization have?", ["", "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
user_details['remote_work'] = st.selectbox("Do you work remotely (outside of an office) at least 50% of the time?", ["", "Yes", "No"])
user_details['tech_company'] = st.selectbox("Is your employer primarily a tech company/organization?", ["", "Yes", "No"])
user_details['benefits'] = st.selectbox("Does your employer provide mental health benefits?", ["", "Yes", "No", "Don't know"])
user_details['care_options'] = st.selectbox("Do you know the options for mental health care your employer provides?", ["", "Yes", "No", "Not sure"])
user_details['wellness_program'] = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", ["", "Yes", "No", "Don't know"])
user_details['seek_help'] = st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", ["", "Yes", "No", "Don't know"])
user_details['anonymity'] = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse?", ["", "Yes", "No", "Don't know"])
user_details['leave'] = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", ["", "Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"])
user_details['comments'] = st.text_input("Any additional notes or comments")


with st.expander('Advanced: supply support score (optional)'):
    st.write('A numeric support score used by the model. If you leave this blank the model will use the training mean.')
    user_details['support_score'] = st.text_input('Support score (numeric, optional)')

if st.button('Predict'):
    model_input = {
        'age': user_details['age'],
        'gender': user_details['gender'] if user_details['gender'] else 'Other',
        'family_history': user_details['family_history'] if user_details['family_history'] else 'No',
        'no_employees': user_details['no_employees'] if user_details['no_employees'] else '1-5',
        'work_interfere': user_details['work_interfere'] if user_details['work_interfere'] else 'Unknown',
        'benefits': user_details['benefits'],
        'care_options': user_details['care_options'],
        'wellness_program': user_details['wellness_program'],
        'support_score': float(user_details['support_score']) if user_details.get('support_score') else None
    }

    result = predict_from_user_details(model_input)
    st.subheader('Prediction Result')
    st.write(f"Predicted label (1 = sought treatment): {result['pred_label']}")
    st.write(f"Probability of seeking treatment: {result['pred_proba']:.3f}")

