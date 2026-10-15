"""Reproducible training from synthetic sample dataset."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
df=pd.read_csv("transactions_synthetic.csv")
X=df.drop(columns=["is_fraud"]); y=df["is_fraud"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
model=RandomForestClassifier(n_estimators=150,max_depth=9,min_samples_leaf=8,class_weight="balanced",random_state=42,n_jobs=-1)
model.fit(X_train,y_train)
prob=model.predict_proba(X_test)[:,1]; pred=(prob>=0.35).astype(int)
metrics={"precision":float(precision_score(y_test,pred,zero_division=0)),"recall":float(recall_score(y_test,pred,zero_division=0)),"f1":float(f1_score(y_test,pred,zero_division=0)),"average_precision":float(average_precision_score(y_test,prob)),"confusion_matrix":confusion_matrix(y_test,pred).tolist(),"threshold":0.35}
joblib.dump(model,"model.joblib")
with open("metrics.json","w") as f: json.dump(metrics,f,indent=2)
print(metrics)
