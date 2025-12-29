from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: str)->pd.DataFrame:
    """
    Loads the dataset.
    """
    return pd.read_csv(file_path)

def eda(df: pd.DataFrame):
    """
    Prints basic information about the dataset.
    """
    print("Dataset shape:",df.shape)
    print("\nColumn info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

def target_imbalance(df: pd.DataFrame,target: str):
    """
    Shows class distribution of the target variable.
    """
    print("\nTarget value counts:")
    print(df[target].value_counts())
    sns.countplot(x=target,data=df)
    plt.title("Target Distribution")
    plt.show()

def numeric_summary(df: pd.DataFrame):
    """
    Displays summary statistics for numeric features.
    """
    print("\nStatistical summary:")
    print(df.describe())

def feature_split(df: pd.DataFrame,target:str):
    """
    Splits the data for target.
    """
    X=df[["principal","terms","age","education","gender"]]
    y=df[target]
    return X,y

def data_split(X,y):
    """
    Splits the data for train and test.
    """
    return train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

def decision_tree(X_train,y_train):
    """
    Trains decision tree.
    """
    model=DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train,y_train)
    return model

def random_forest(X_train,y_train):
    """
    Trains random forest.
    """
    model=RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train,y_train)
    return model

def model_evaluation(model,X_test,y_test):
    """
    Displays performance measures of trained model.
    """
    pred=model.predict(X_test)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))

def save_model(model,model_name):
    """
    Saves the trained model.
    """
    with open(model_name,"wb") as f:
        pickle.dump(model,f)

if __name__=="__main__":
    df=load_data("loan.csv")
    df.columns=df.columns.str.strip().str.lower()
    df=df.apply(lambda x:x.strip().lower() if isinstance(x,str) else x)

    df["loan_status"]=df["loan_status"].replace({
        "paidoff":1,
        "paid off":1,
        "approved":1,
        "collection":0,
        "collection_paidoff":0,
        "rejected":0
    })
    df=df.dropna(subset=["loan_status"])

    X,y=feature_split(df,"loan_status")
    X=pd.get_dummies(X,drop_first=True)

    with open("feature_columns.pkl","wb") as f:
        pickle.dump(X.columns.tolist(),f)

    X_train,X_test,y_train,y_test=data_split(X,y)

    print("\nDecision Tree")
    dt_model=decision_tree(X_train,y_train)
    model_evaluation(dt_model,X_test,y_test)

    print("\nRandom Forest")
    rf_model=random_forest(X_train,y_train)
    model_evaluation(rf_model,X_test,y_test)

    save_model(dt_model,"decision_tree.pkl")
    save_model(rf_model,"random_forest.pkl")
