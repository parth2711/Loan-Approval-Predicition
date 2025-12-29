import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Prediction",page_icon="🏦",layout="centered")

#loading models
with open("decision_tree.pkl","rb") as f:
    dt_model=pickle.load(f)
with open("random_forest.pkl","rb") as f:
    rf_model=pickle.load(f)
with open("feature_columns.pkl","rb") as f:
    feature_columns=pickle.load(f)

#proceeding with ui
st.title("🏦 Loan Approval Prediction")
st.caption("Decision Tree and Random Forest based system")

st.divider()

with st.expander("ℹ️ About this app"):
    st.write("""
    This web app predicts whether a loan application is likely to be Approved or Rejected.
    Models used: Decision Tree and Random Forest.
    """)

st.subheader("Model Selection")
model_choice=st.radio("Choose a model",["Decision Tree","Random Forest"],horizontal=True)
model=dt_model if model_choice=="Decision Tree" else rf_model

st.divider()
st.subheader("Applicant Details")

col1,col2=st.columns(2)
with col1:
    gender=st.selectbox("Gender",["Male","Female"])
    married=st.selectbox("Married",["Yes","No"])
    education=st.selectbox("Education",["Graduate","Not Graduate"])
    dependents=st.selectbox("Dependents",["0","1","2","3+"])
    applicant_income=st.number_input("Applicant Income",min_value=0,max_value=10000000,value=5000000)
    loan_amount=st.number_input("Loan Amount",min_value=0,max_value=50000000,value=10000000)
with col2:
    self_employed=st.selectbox("Self Employed",["Yes","No"])
    property_area=st.selectbox("Property Area",["Urban","Semiurban","Rural"])
    credit_history=st.selectbox("Credit History",[1.0,0.0])
    coapplicant_income=st.number_input("Coapplicant Income",min_value=0,max_value=10000000,value=0)
    loan_term=st.number_input("Loan Term (months)",min_value=1,max_value=480,value=360)

# Create input dictionary with LOWERCASE keys to match main.py
input_dict={
    "gender":gender,
    "married":married,
    "education":education,
    "self_employed":self_employed,
    "property_area":property_area,
    "dependents":dependents,
    "applicantincome":applicant_income,
    "coapplicantincome":coapplicant_income,
    "loanamount":loan_amount,
    "loan_amount_term":loan_term,
    "credit_history":credit_history
}

input_df=pd.DataFrame([input_dict])
input_df=pd.get_dummies(input_df,drop_first=True)

# CRITICAL FIX: Align input columns with training columns
input_df=input_df.reindex(columns=feature_columns,fill_value=0)

st.divider()

if st.button("Predict Loan Status"):
    pred=model.predict(input_df)[0]
    prob=model.predict_proba(input_df)[0]
    if pred==1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
    st.subheader("Prediction Confidence")
    st.write(f"Approval Probability: {prob[1]:.2%}")
    st.write(f"Rejection Probability: {prob[0]:.2%}")
    st.progress(int(prob[1]*100))
    st.caption(f"Prediction made using {model_choice}")

st.divider()

with st.expander("About me"):
    st.write("""
    Hi! I'm Parth, a Computer Science student learning Machine Learning.
    """)