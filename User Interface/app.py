import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scalers, and feature names
rf_model = joblib.load('rf_model.pkl')
scaler_time = joblib.load('scaler_time.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')
feature_order = joblib.load('feature_names.pkl')  # Load feature names from training

# Compute V1-V28 means from data
data = pd.read_csv('creditcard.csv')
data['scaled_time'] = scaler_time.transform(data[['Time']])
data['scaled_amount'] = scaler_amount.transform(data[['Amount']])
data = data.drop(['Time', 'Amount', 'Class'], axis=1)
v_means = data.mean().to_dict()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details for prediction.")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
time_hours = st.number_input("Time (hours since midnight)", min_value=0.0, max_value=23.99, value=12.0, step=0.01)
time_seconds = time_hours * 3600

if st.button("Predict"):
    # Prepare input data with correct feature order
    scaled_time = scaler_time.transform([[time_seconds]])[0][0]
    scaled_amount = scaler_amount.transform([[amount]])[0][0]
    
    # Create input dictionary with all features
    input_dict = {'scaled_time': scaled_time, 'scaled_amount': scaled_amount}
    for feature in feature_order:
        if feature not in input_dict:
            input_dict[feature] = v_means.get(feature, 0.0)  # Use mean for V1-V28
    
    # Create DataFrame with correct feature order
    input_data = pd.DataFrame([input_dict], columns=feature_order)
    
    # Predict
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0]
    result = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    st.write(f"Transaction is: **{result}**")
    st.write(f"Fraud Probability: **{probability[1]:.2%}**")