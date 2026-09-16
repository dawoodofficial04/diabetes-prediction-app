import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained model (a single pipeline: median-imputer + scaler + logistic regression)
@st.cache_resource
def load_model():
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# These columns can't medically be 0 (e.g. BMI of 0), so a 0 here means the
# value is missing, not that it's actually zero. The model was trained the
# same way, so this mapping must match at inference time.
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

st.title("Diabetes Prediction App")
st.write("Enter the following details to predict the likelihood of diabetes:")
st.divider()

# Input fields for user input
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
st.caption("Tip: if you don't know a value (e.g. Insulin or Skin Thickness), leave it at 0 — "
           "the model treats 0 in these fields as \"unknown\" and estimates it instead of "
           "treating it as a real zero.")
st.divider()

# Button to predict diabetes
if st.button("Predict Diabetes"):
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    # Same "0 means missing" convention used at training time
    input_df[ZERO_AS_MISSING] = input_df[ZERO_AS_MISSING].replace(0, np.nan)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of diabetes

    if prediction == 1:
        st.error(f"Result: Diabetic (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"Result: Non Diabetic (Confidence: {(1-probability)*100:.2f}%)")

    st.caption(
        "This tool flags risk based on patterns in historical data and is not a medical "
        "diagnosis. Please consult a healthcare professional for an accurate assessment."
    )