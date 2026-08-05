# Customer Churn Prediction

A trained scikit-learn model that predicts whether a telecom customer is likely to churn, based on their account and service details.

## Model Summary

| | |
|---|---|
| **Algorithm** | Random Forest Classifier (100 trees) |
| **Pipeline** | `ColumnTransformer` (StandardScaler + OneHotEncoder) → `RandomForestClassifier` |
| **Target** | `Churn` — `Yes` / `No` |
| **Features** | 19 raw inputs (4 numeric, 15 categorical) → 45 encoded features |
| **scikit-learn version** | 1.6.1 (**must** match — see Compatibility note below) |

### Input features

**Numeric:** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`

**Categorical:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

### Top predictors (feature importance)

1. `TotalCharges`
2. `tenure`
3. `MonthlyCharges`
4. `Contract = Month-to-month`
5. `PaymentMethod = Electronic check`
6. `OnlineSecurity = No`
7. `InternetService = Fiber optic`

Customers with short tenure, month-to-month contracts, and electronic-check payments are the strongest churn signals — the classic "not locked in, low switching cost" profile.

### ⚠️ Compatibility note

This model was pickled with **scikit-learn 1.6.1**. Loading it with a newer/older scikit-learn version can raise errors like `_RemainderColsList` or fail silently with different results. `requirements.txt` pins the exact version — install from it rather than the latest scikit-learn.

## Project Structure

```
churn-prediction/
├── model/
│   └── customer_churn_model.pkl   # trained pipeline
├── app.py                          # Streamlit web UI
├── predict.py                      # CLI batch prediction script
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Option 1 — Web app (interactive)

```bash
streamlit run app.py
```
Opens a browser form where you fill in customer details and get an instant churn prediction + probability.

### Option 2 — Batch predictions from CSV

```bash
python predict.py --input new_customers.csv --output predictions.csv
```

Your input CSV must contain these columns:
`SeniorCitizen, tenure, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod`

### Option 3 — Use it in your own code

```python
import joblib
import pandas as pd

model = joblib.load("model/customer_churn_model.pkl")

customer = pd.DataFrame([{
    "SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
    "gender": "Male", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
}])

prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]  # probability of "Yes"
print(prediction, probability)
```

## Deployment (optional)

Deploy the Streamlit app for free at [share.streamlit.io](https://share.streamlit.io):
1. Push this repo to GitHub (steps below).
2. Sign in to Streamlit Community Cloud with your GitHub account.
3. Click **New app**, select this repo, branch `main`, and file `app.py`.
4. Deploy — you'll get a public URL to share.

## License

Add a license of your choice (MIT is a common default for portfolio projects).
