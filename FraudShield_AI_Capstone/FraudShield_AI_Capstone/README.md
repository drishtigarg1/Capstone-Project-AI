# FraudShield AI — AI-Powered Transaction Fraud Detection

## Problem
Financial transaction reviewers need help prioritizing suspicious transactions. This educational prototype uses a Random Forest classifier to flag transactions for human review using synthetic transaction records.

## Data and approach
- **6,000 fully synthetic records**, created for demonstration; no customer or banking records used.
- Features: amount, hour, hourly transaction count, international flag, new-device flag, account age.
- Stratified 75/25 train/test split, random seed 42.
- Random Forest with class balancing, threshold 0.35; precision, recall, F1, average precision and confusion matrix.
- No claims of real-world predictive validity. Labels were simulated from feature-dependent probabilities, so results partly reflect synthetic generation assumptions.

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Demonstration metrics (synthetic held-out set)
Precision: 9.9%; Recall: 53.6%; F1: 16.7%; Average precision: 11.1%. Threshold: 0.35.

## Project structure
`app.py` Streamlit interface; `train.py` training pipeline; `transactions_synthetic.csv` synthetic data; `model.joblib` trained model; `metrics.json` results; `executive_summary.pdf` report; `five_slide_pitch.pptx` pitch.

## Responsible AI and limitations
Scores should only trigger review, never automatic account suspension. Evaluate on representative permissioned data before deployment, calibrate probabilities, check subgroup fairness, monitor drift and false positives, and establish human appeal paths. The demo does not ingest personal identifiers.

## Future scope
Replace synthetic records with permissioned anonymized transactions, time-based evaluation, model calibration, operational alerting, and drift monitoring.
# Capstone-Project-AI
