import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- Load model data ----------------
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
gender_encoder = data["gender_encoder"]
subsType_encoder = data["subsType_encoder"]
contract_encoder = data["contract_encoder"]
feature_columns = data["feature_columns"]

st.title("Customer Churn Prediction App")
st.header("Enter Customer Details")

# ---------------- Numeric Inputs ----------------
age = st.number_input("Age", min_value=0, max_value=120, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
usage_frequency = st.number_input("Usage Frequency", min_value=0, value=5)
support_calls = st.number_input("Support Calls", min_value=0, value=1)
payment_delays = st.number_input("Payment Delays", min_value=0, value=0)
total_spent = st.number_input("Total Spent", min_value=0, value=1000)
last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, value=10)

# ---------------- Categorical Inputs ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

# ---------------- Predict Button ----------------
if st.button("Predict Churn"):

    # Encode categorical values
    gender_encoded = gender_encoder.transform([gender])[0]
    subscription_encoded = subsType_encoder.transform([subscription_type])[0]
    contract_encoded = contract_encoder.transform([contract_length])[0]

    # Create input dataframe (MUST match training columns)
    input_df = pd.DataFrame([[
        age,
        gender_encoded,
        tenure,
        usage_frequency,
        support_calls,
        payment_delays,
        subscription_encoded,
        contract_encoded,
        total_spent,
        last_interaction
    ]], columns=feature_columns)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Output
    if prediction == 1:
        st.error(
            f"Customer is likely to churn ⚠️ "
            f"(Probability: {prediction_proba[1] * 100:.2f}%)"
        )
    else:
        st.success(
            f"Customer will not churn ✅ "
            f"(Probability: {prediction_proba[0] * 100:.2f}%)"
        )
