import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

if __name__=="__main__":
    dataset=pd.read_csv("cleaned_data.csv")
    X=dataset.drop(columns=['Loan_Status'])
    y=dataset['Loan_Status']
    X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.2)
    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()
        model=ElasticNet()
        model.fit(X_train,y_train)

    
