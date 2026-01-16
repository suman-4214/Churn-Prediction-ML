# scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'MonthlyCharges', 'Tenure','Gender'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

st.divider()

st.write("Please provide the following details to predict customer churn:")

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter tenure (in months)", min_value=0, max_value=130, value=12)

monthly_charges = st.number_input("Enter monthly charges", min_value=20, max_value=150)

gender = st.selectbox("Enter the gender",["Male","Female"])

st.divider()

predict_button = st.button("Predict Churn")

st.divider()

if predict_button:
    gender_selected = 1 if gender== 'Female' else 0

    X = [age, monthly_charges, tenure, gender_selected]

    X1 = np.array(X)
    
    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()

    st.write(f"The predicted customer status is: {predicted}")

else:
    st.write("Please enter the required details to predict customer churn.")