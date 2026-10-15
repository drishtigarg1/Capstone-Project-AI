import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
st.set_page_config(page_title="FraudShield AI",page_icon="🛡️",layout="wide")
st.title("🛡️ FraudShield AI")
st.caption("Educational fraud-risk demo • synthetic training data • not for financial decisions")
root=Path(__file__).parent
model=joblib.load(root/"model.joblib")
metrics=json.loads((root/"metrics.json").read_text())
left,right=st.columns([2,1])
with left:
 st.subheader("Assess a transaction")
 with st.form("transaction"):
  amount=st.number_input("Amount",min_value=1.0,value=120.0)
  hour=st.slider("Transaction hour (0–23)",0,23,14)
  velocity=st.number_input("Transactions in past hour",min_value=1,value=2,step=1)
  international=st.checkbox("International transaction")
  new_device=st.checkbox("New device")
  age=st.number_input("Account age in days",min_value=1,value=400,step=1)
  submit=st.form_submit_button("Estimate risk")
 if submit:
  sample=pd.DataFrame([{"amount":amount,"hour":hour,"transactions_last_hour":velocity,"international":int(international),"new_device":int(new_device),"account_age_days":age}])
  risk=float(model.predict_proba(sample)[0,1])
  st.metric("Estimated risk score",f"{risk:.1%}")
  st.warning("Flag for human review" if risk>=metrics["threshold"] else "Below review threshold")
  st.caption("This is a model score, not a verified probability of real-world fraud.")
with right:
 st.subheader("Held-out synthetic test results")
 for k in ["precision","recall","f1","average_precision"]: st.metric(k.replace("_"," ").title(),f"{metrics[k]:.1%}")
 st.caption("Threshold: 35%. These figures do not generalize to real transactions.")
st.divider()
st.subheader("Dataset overview")
df=pd.read_csv(root/"transactions_synthetic.csv")
st.write(f"{len(df):,} synthetic transactions; {df.is_fraud.mean():.1%} simulated fraud labels")
st.bar_chart(df.groupby("hour")["is_fraud"].mean().rename("Simulated fraud rate"))
st.dataframe(df.head(12),use_container_width=True)
