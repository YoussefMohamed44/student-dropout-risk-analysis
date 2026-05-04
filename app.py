import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('student_dropout_model.joblib')
feature_names = joblib.load('feature_names.joblib')

st.set_page_config(page_title="Dropout Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Risk Analysis")

st.info("Input student metrics to determine dropout probability and risk level.")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sem1_grade = st.number_input("1st Sem Grade", 0.0, 20.0, 10.0)
        sem2_grade = st.number_input("2nd Sem Grade", 0.0, 20.0, 10.0)
        sem2_appr = st.number_input("Units Approved (2nd Sem)", 0, 20, 5)
        
    with col2:
        tuition = st.selectbox("Tuition Fees Up to Date?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        scholarship = st.selectbox("Scholarship?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        debtor = st.selectbox("Debtor?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    submit = st.form_submit_button("Analyze Risk")

if submit:
    feature_names = joblib.load('feature_names.joblib')
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    input_df['Curricular units 1st sem (grade)'] = sem1_grade
    input_df['Curricular units 2nd sem (grade)'] = sem2_grade
    input_df['Curricular units 2nd sem (approved)'] = sem2_appr
    input_df['Tuition fees up to date'] = tuition
    input_df['Scholarship holder'] = scholarship
    input_df['Debtor'] = debtor
    
    prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if prob > 0.5:
        st.error(f"High Risk Detected: {prob:.1%}")
        st.progress(float(prob))
    else:
        st.success(f"Low Risk: {1-prob:.1%} safety score")
        st.progress(float(prob))