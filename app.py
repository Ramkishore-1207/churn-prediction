"""
Streamlit web app for the customer churn prediction model.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud by pointing it at this file
in your GitHub repo (see README.md).
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/customer_churn_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
    st.title("📉 Customer Churn Prediction")
    st.write("Enter a customer's details to predict whether they are likely to churn.")

    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

    if st.button("Predict Churn", type="primary"):
        input_df = pd.DataFrame([{
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][list(model.classes_).index("Yes")]

        if prediction == "Yes":
            st.error(f"⚠️ Likely to churn — probability: {proba:.1%}")
        else:
            st.success(f"✅ Likely to stay — churn probability: {proba:.1%}")


if __name__ == "__main__":
    main()
