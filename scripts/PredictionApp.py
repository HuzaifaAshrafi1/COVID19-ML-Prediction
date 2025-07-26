import streamlit as st
import joblib
import numpy as np
import os

# Set Streamlit page config
st.set_page_config(page_title="COVID-19 Risk Predictor", layout="centered")
st.title("COVID-19 Death Risk Predictor")
st.write("Choose symptoms and a model to predict the risk using regression or classification.")

# Define symptom features
symptom_features = [
    'fever', 'dry cough', 'fatigue', 'loss of taste',
    'loss of or smell', 'difficulty breathing', 'sore throat',
    'headache', 'muscle aches', 'chills', 'diarrhea',
    'runny nose', 'nausea'
]

# Model paths
regression_models = {
    "Polynomial Regression": "models/polynomial_regression_model.pkl",
    "Linear Regression": "models/linear_regression_model.pkl"
}

classification_models = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "K-Nearest Neighbors (KNN)": "models/knn_model.pkl",
    "Naive Bayes": "models/naive_bayes_model.pkl"
}

# Sidebar model selection
st.sidebar.header("Choose Prediction Type & Model")
mode = st.sidebar.radio("Prediction Type", ["Regression", "Classification"])

if mode == "Regression":
    model_name = st.sidebar.selectbox(
        "Choose Regression Model",
        list(regression_models.keys()),
        index=list(regression_models.keys()).index("Linear Regression")  # Default selection
    )
    model_path = regression_models[model_name]
else:
    model_name = st.sidebar.selectbox("Choose Classification Model", list(classification_models.keys()))
    model_path = classification_models[model_name]

# Load selected model
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)

# Input section
user_input = []

# Only show Total Cases input for Polynomial Regression
if mode == "Regression" and model_name == "Polynomial Regression":
    total_cases = st.number_input("Total Cases per Million", min_value=0.0, value=5000.0, step=100.0)
    user_input.append(total_cases)

st.subheader("Symptom Selection (1 = Yes, 0 = No)")
for feature in symptom_features:
    val = st.selectbox(f"{feature.title()}:", [0, 1], key=feature + mode + model_name)
    user_input.append(val)

# Create input array
input_array = np.array([user_input])

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_array)[0]

    if mode == "Regression":
        st.write(f"**Predicted Deaths per Million:** `{prediction:.2f}`")

        # Risk interpretation
        if prediction >= 11.67:
            st.error("High Risk of Death")
        elif 5 < prediction < 11.67:
            st.warning("Moderate Risk of Death")
        else:
            st.success("Low Risk of Death")

    else:
        # Classification output
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_array)[0][1]
            st.write(f"Prediction Probability (High Risk): `{prob:.2f}`")

        if prediction == 1:
            st.error("High Death Risk Detected")
        else:
            st.success("Low Death Risk")
