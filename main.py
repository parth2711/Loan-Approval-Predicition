from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: str)->pd.DataFrame:
    return pd.read_csv(file_path)
    
def feature_split(df: pd.DataFrame,target:str):
    X=df.drop(target,axis=1)
    y=df[target]
    return X,y

def data_split(X,y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def decision_tree(X_train,y_train):
    dt_model=DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train,y_train)
    return dt_model

def random_forest(X_train,y_train):
    rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
    rf_model.fit(X_train,y_train)
    return rf_model

def model_evaluation(model,X_test,y_test):
    prediction=model.predict(X_test)
    print(confusion_matrix(y_test,prediction))
    print(classification_report(y_test,prediction))

def save_model(model,model_name):
    with open(model_name,'wb') as f:
        pickle.dump(model,f)

if __name__=="__main__":
    df=load_data("loan_approval_dataset.csv")
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df[" loan_status"]=df[" loan_status"].map({'Approved':1,'Rejected':0})
    
    X,y=feature_split(df," loan_status")

    X=pd.get_dummies(X,drop_first=True)

    X_train, X_test, y_train, y_test=data_split(X,y)

    print("\nDecision Tree")
    dt_model=decision_tree(X_train,y_train)
    model_evaluation(dt_model,X_test,y_test)
    
    print("\nRandom Forest")
    rf_model=random_forest(X_train,y_train)
    model_evaluation(rf_model,X_test,y_test)