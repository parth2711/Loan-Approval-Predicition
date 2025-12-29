import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Prediction",page_icon="🏦",layout="centered")

with open("decision_tree.pkl","rb") as f:
    dt_model=pickle.load(f)
with open("random_forest.pkl","rb") as f:
    rf_model=pickle.load(f)
with open("encoder.pkl","rb") as f:
    encoder=pickle.load(f)
with open("feature_columns.pkl","rb") as f:
    feature_columns=pickle.load(f)

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
    education=st.selectbox("Education",["Graduate","Not Graduate"])
    self_employed=st.selectbox("Self Employed",["Yes","No"])
    no_of_dependents=st.number_input("Number of Dependents",min_value=0,max_value=10,value=2)
    income_annum=st.number_input("Annual Income",min_value=10000,value=5000000)
    loan_amount=st.number_input("Loan Amount",min_value=10000,value=2000000)
    loan_term=st.number_input("Loan Term (Years)",min_value=1,max_value=40,value=10)
with col2:
    cibil_score=st.number_input("CIBIL Score",min_value=300,max_value=900,value=700)
    residential_assets_value=st.number_input("Residential Assets Value",value=1000000)
    commercial_assets_value=st.number_input("Commercial Assets Value",value=500000)
    luxury_assets_value=st.number_input("Luxury Assets Value",value=1000000)
    bank_asset_value=st.number_input("Bank Asset Value",value=500000)

input_dict={
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

st.divider()

if st.button("Predict Loan Status"):
    input_df=pd.DataFrame([input_dict])
    
    cat_cols=['education','self_employed']
    num_cols=[c for c in input_df.columns if c not in cat_cols]
    
    encoded_cats=encoder.transform(input_df[cat_cols])
    encoded_df=pd.DataFrame(encoded_cats,columns=encoder.get_feature_names_out(cat_cols))
    
    input_final=pd.concat([encoded_df,input_df[num_cols].reset_index(drop=True)],axis=1)
    input_final=input_final.reindex(columns=feature_columns,fill_value=0)

    pred=model.predict(input_final)[0]
    prob=model.predict_proba(input_final)[0]
    
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
    Hi! I'm Parth, a Computer Science student learning Machine Learning
    by building small, end-to-end applications.
    """)