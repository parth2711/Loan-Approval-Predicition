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

label_map={
    0:"COLLECTION",
    1:"PAIDOFF",
    2:"COLLECTION_PAIDOFF"
}

#proceeding with ui
st.title("🏦 Loan Approval Prediction")
st.caption("Decision Tree and Random Forest based system")

st.divider()

with st.expander("ℹ️ About this app"):
    st.write("""
    This web app predicts the repayment status of a loan application.

    Models used:
    - Decision Tree
    - Random Forest

    This app was built as part of my learning journey to understand
    tree-based models on real-world tabular data.
    """)

st.subheader("Model Selection")
model_choice=st.radio("Choose a model",["Decision Tree","Random Forest"],horizontal=True)
model=dt_model if model_choice=="Decision Tree" else rf_model

st.divider()
st.subheader("Applicant Details")

col1,col2=st.columns(2)
with col1:
    gender=st.selectbox("Gender",["male","female"])
    education=st.selectbox("Education",["high school or below","college","bachelor","master or above"])
    age=st.number_input("Age",min_value=18,max_value=100,value=30)
with col2:
    principal=st.number_input("Principal Amount",min_value=300,max_value=1000,value=1000,step=100)
    terms=st.selectbox("Terms (days)",[7,15,30])

input_dict={
    "gender":gender.lower(),
    "education":education.lower(),
    "age":age,
    "principal":principal,
    "terms":terms
}

input_df=pd.DataFrame([input_dict])
input_df=pd.get_dummies(input_df,drop_first=True)
input_df=input_df.reindex(columns=feature_columns,fill_value=0)

st.divider()

if st.button("Predict Loan Status"):
    pred=model.predict(input_df)[0]
    probs=model.predict_proba(input_df)[0]

    pred_label=label_map[pred]

    if pred_label=="PAIDOFF":
        st.success("Loan Likely to be Paid Off")
    elif pred_label=="COLLECTION_PAIDOFF":
        st.warning("Loan Paid After Collection")
    else:
        st.error("Loan Likely to go into Collection")

    st.subheader("Prediction Confidence")
    for i,label in label_map.items():
        st.write(f"{label}: {probs[i]:.2%}")

    st.progress(int(probs[pred]*100))
    st.caption(f"Prediction made using {model_choice}")

st.divider()

with st.expander("About me"):
    st.write("""
    Hi! I'm Parth, a Computer Science student learning Machine Learning
    by building small, end-to-end applications.

    I built this app to focus on understanding model behavior.
    For now, this marks the end of my core ML model learning phase,
    and I will move towards DSA and system design while continuing ML in the future.
    """)
