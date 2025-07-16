import streamlit as st
import joblib
import numpy as np

# To run this streamlit paste below code in the terminal
# python -m streamlit run app.py

# Load the trained model
model = joblib.load("knn_model.pkl")  # Change filename as needed

st.title("Corona Death Risk Predictor")

st.write("Please select symptom presence (1 = Yes, 0 = No):")

# List of symptoms 
symptoms = [
    "fever", "dry cough", "fatigue", "loss of taste",
    "loss of or smell", "difficulty breathing", "sore throat",
    "headache", "muscle aches", "chills", "diarrhea",
    "runny nose", "nausea"
]
 
# Create input fields for each symptom
user_input = []
for symptom in symptoms:
    val = st.selectbox(f"{symptom.title()}:", [0, 1], key=symptom)
    user_input.append(val)

# Predict button
if st.button("Predict Death Risk"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]

    if prediction >=11.67:
        st.error("High Risk of Death")
    else:
        st.success("Low Risk of Death")
