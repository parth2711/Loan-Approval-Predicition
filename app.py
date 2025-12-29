# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# st.set_page_config(
#     page_title="Loan Approval Prediction",
#     page_icon="🏦",
#     layout="centered"
# )

# #loading models
# with open("decision_tree.pkl", "rb") as f:
#     dt_model = pickle.load(f)

# with open("random_forest.pkl", "rb") as f:
#     rf_model = pickle.load(f)

# with open("feature_columns.pkl", "rb") as f:
#     feature_columns = pickle.load(f)

# #loading reference dataset to align columns
# df_ref = pd.read_csv("loan_approval_dataset.csv")
# df_ref.columns = df_ref.columns.str.strip()
# df_ref = df_ref.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# df_ref["loan_status"] = df_ref["loan_status"].map({"Approved": 1, "Rejected": 0})
# X_ref = pd.get_dummies(df_ref.drop("loan_status", axis=1), drop_first=True)

# #proceeding with ui
# st.title("🏦 Loan Approval Prediction")
# st.caption("Decision Tree & Random Forest based prediction system")

# st.divider()

# with st.expander("ℹ️ About this app"):
#     st.write(
        # """
        # This web app predicts whether a loan application is likely to be **approved or rejected**.

        # **Models used:**
        # - Decision Tree
        # - Random Forest

        # This app was built as part of my learning journey to understand how
        # tree-based models work on real-world tabular data.
        # """
#     )

# with st.expander("👨‍💻 About me"):
#     st.write(
#         """
#         Hi! I'm **Parth**, a Computer Science student learning Machine Learning
#         by building small, end-to-end applications.

#         This app focuses more on **understanding model behavior**
#         than chasing perfect accuracy.
#         """
#     )

# st.subheader("🔍 Choose a model")
# model_choice = st.radio("Select the model you want to use:",["Decision Tree","Random Forest"])

# model=dt_model if model_choice=="Decision Tree" else rf_model

# st.divider()

# st.subheader("📝 Enter applicant details")

# gender=st.selectbox("Gender",["Male","Female"])
# married=st.selectbox("Married",["Yes","No"])
# education=st.selectbox("Education",["Graduate","Not Graduate"])
# self_employed=st.selectbox("Self Employed",["Yes","No"])
# property_area=st.selectbox("Property Area",["Urban","Semiurban","Rural"])

# dependents=st.selectbox("Dependents",["0","1","2","3+"])

# applicant_income=st.number_input("Applicant Income",min_value=0)
# coapplicant_income=st.number_input("Coapplicant Income",min_value=0)
# loan_amount=st.number_input("Loan Amount",min_value=0)
# loan_term=st.number_input("Loan Term (months)",min_value=0)
# credit_history=st.selectbox("Credit History",[1.0, 0.0])

# input_dict = {
#     "gender": gender,
#     "married": married,
#     "education": education,
#     "self_employed": self_employed,
#     "property_area": property_area,
#     "dependents": dependents,
#     "applicantincome": applicant_income,
#     "coapplicantincome": coapplicant_income,
#     "loanamount": loan_amount,
#     "loan_amount_term": loan_term,
#     "credit_history": credit_history
# }

# input_df=pd.DataFrame([input_dict])

# input_df=pd.get_dummies(input_df,drop_first=True)

# input_df=input_df.reindex(columns=feature_columns,fill_value=0)

# st.divider()

# if st.button("🔮 Predict Loan Status"):
#     prediction=model.predict(input_df)[0]

#     if prediction==1:
#         st.success("✅ Loan Approved")
#     else:
#         st.error("❌ Loan Rejected")

#     st.caption(f"Prediction made using **{model_choice}** model")

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Prediction",page_icon="🏦",layout="centered")

with open("decision_tree.pkl","rb") as f:
    dt_model=pickle.load(f)
with open("random_forest.pkl","rb") as f:
    rf_model=pickle.load(f)
with open("feature_columns.pkl","rb") as f:
    feature_columns=pickle.load(f)

st.title("🏦 Loan Approval Prediction")
st.caption("Decision Tree and Random Forest based system")

st.divider()

with st.expander("ℹ️ About this app"):
    st.write("""
        This web app predicts whether a loan application is likely to be **Approved or Rejected**.

        **Models used:**
        - Decision Tree
        - Random Forest

        This app was built as part of my learning journey to understand how
        tree-based models work on real-world tabular data.
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
    applicant_income=st.number_input("Applicant Income",min_value=0)
    loan_amount=st.number_input("Loan Amount",min_value=0)
with col2:
    self_employed=st.selectbox("Self Employed",["Yes","No"])
    property_area=st.selectbox("Property Area",["Urban","Semiurban","Rural"])
    credit_history=st.selectbox("Credit History",[1.0,0.0])
    coapplicant_income=st.number_input("Coapplicant Income",min_value=0)
    loan_term=st.number_input("Loan Term (months)",min_value=0)

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
input_df=input_df.reindex(columns=feature_columns,fill_value=0)

st.divider()

if st.button("Predict Loan Status"):
    pred=model.predict(input_df)[0]
    prob=model.predict_proba(input_df)[0]
    approve_prob=prob[1]
    reject_prob=prob[0]

    if pred==1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.subheader("Prediction Confidence")
    st.write(f"Approval Probability: {approve_prob:.2%}")
    st.write(f"Rejection Probability: {reject_prob:.2%}")

    st.progress(int(approve_prob*100))
    st.caption(f"Prediction made using {model_choice}")

st.divider()

with st.expander("About me"):
    st.write("""
            Hi! I'm **Parth**, a Computer Science student learning Machine Learning
            by building small, end-to-end applications.

            I built this app to focus on **understanding model behavior**.
            For now, this marks the end of my learning phase for core ML models,
            and I plan to build more advanced systems in the future as I continue learning.
            """)