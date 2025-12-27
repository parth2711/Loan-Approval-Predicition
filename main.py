from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: pd.DataFrame)->pd.DataFrame:
    df=pd.read_csv(file_path)
    return df

def data_scaling(X: pd.DataFrame):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled

def data_split(X_scaled,y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def decision_tree(X_train):
    dt_model=DecisionTreeClassifier()
    dt_model.fit(X_train)
    return dt_model