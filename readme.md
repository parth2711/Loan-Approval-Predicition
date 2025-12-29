# Loan Approval Prediction

This repository contains a simple **Loan Approval Prediction** system built using **Decision Tree** and **Random Forest** models.

The main goal of this work was to understand how tree-based models behave on real-world tabular data and how a trained model can be used inside a small web app.
This is a **learning-focused build**, not something meant for real financial decisions.

---

## What this does

* Takes applicant details like income, education, credit history, etc.
* Converts categorical data into numerical form using **one-hot encoding**
* Trains two models:

  * Decision Tree
  * Random Forest
* Allows selecting a model and predicting:

  * Loan Approved
  * Loan Rejected
* Shows prediction probabilities for better understanding

---

## Dataset

* Loan Approval Dataset (Kaggle-style loan dataset)
* Contains applicant details and loan status
* Includes both numerical and categorical features
* Required cleaning due to:

  * Extra spaces in column names
  * Categorical values
  * Missing values

---

## Machine Learning Workflow

1. Loaded and cleaned the dataset
2. Removed extra spaces and handled missing values
3. Converted categorical columns
4. Split data into training and testing sets
5. Trained:

   * Decision Tree Classifier
   * Random Forest Classifier
6. Evaluated models using:

   * Confusion Matrix
   * Classification Report
7. Built a Streamlit app for interaction and prediction

---

## Web App Features

* Choose between **Decision Tree** and **Random Forest**
* Enter applicant details through a simple UI
* Get:

  * Approval / Rejection result
  * Prediction probabilities
* Designed to be simple and easy to understand

---

## Tech Stack

* Python
* Pandas
* NumPy
* scikit-learn
* Streamlit

---

## Note

* This is made for **learning purposes only**
* This project had been diverted to another dataset to test a probable bug, everything is back to normal, reach out on linkedin to report any errors! Thankyou :)