---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)
![License](https://img.shields.io/github/license/HuzaifaAshrafi1/COVID19-ML-Prediction)

# COVID-19 Death Risk Predictor

A comprehensive machine learning project analyzing COVID-19 data, building predictive models, and deploying a Streamlit web app to estimate death risk based on user symptoms.

---

## Overview

This project leverages data analysis, visualization, and machine learning (regression & classification) to predict COVID-19 death risks per country and based on individual inputs.

**Key features:**

* Symptom-based risk prediction
* Time & location-based feature engineering
* Multiple regression and classification models
* Interactive Streamlit app for live predictions
* PCA and outlier handling for improved accuracy

---

## Project Structure

```
covid19-risk-predictor/
│
├── data/
│   ├── Actual Data Of Covid-19.csv
│   └── Cleaned Data of Covid-19.csv
│
├── models/
│   ├── linear_regression_model.pkl
│   ├── polynomial_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   └── logistic_regression_model.pkl
│
├── scripts/
│   ├── PredictionApp.py                     # Streamlit Web App
│   └── COVID 19 Analysis,ML & Models.ipynb # Data Analysis & Modeling Notebook
│
├── visuals/
│   └── *.jpg (Plots, PCA, Boxplots, Confusion Matrix, etc.)
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Models

### Regression

* Linear Regression
* Polynomial Regression (degree=2)

### Classification

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Naive Bayes

---

## Visualizations

* Word Cloud (Symptom Frequency)
* Distributions (Histograms, KDEs)
* PCA Plot (2D projection)
* Boxplots (Outliers)
* Confusion Matrices
* ROC-AUC Curves

---

## Web Application (Streamlit)

### Features

* Choose between Regression or Classification prediction
* Select the prediction model
* Input symptoms manually (1 = Yes, 0 = No)
* For regression: input total COVID-19 cases per million
* Receive real-time risk predictions with intuitive interpretations

### Run the app locally

```bash
python -m streamlit run scripts/PredictionApp.py
```

---

## Installation

Ensure you have Python 3.7+ installed, then run:

```bash
# Clone the repository
git clone https://github.com/HuzaifaAshrafi1/COVID19-ML-Prediction.git

# Navigate to project folder
cd COVID19-ML-Prediction

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run scripts/PredictionApp.py
```

---

## Requirements

Listed in `requirements.txt`:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
imblearn
joblib
wordcloud
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Data Cleaning Process

* Handling missing values
* Encoding categorical variables
* Creating time-based features (year, month, day)
* Treating outliers using IQR & Winsorization
* Symptom flagging & feature engineering

---

## Model Evaluation Metrics

### Regression:

* R² Score
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Sum of Squared Errors (SSE)

### Classification:

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* ROC-AUC Curves

---

## Visual Outputs

All plots, PCA visuals, and confusion matrices are saved in:

```
visuals/
```

---

## Deployment

Deploy your Streamlit app easily on **Streamlit Community Cloud**:

1. Push your complete project to GitHub, including:

   * `scripts/PredictionApp.py`
   * `models/` folder with model `.pkl` files
   * `requirements.txt`
2. Sign up/log in at [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Create a new app linked to your GitHub repo
4. Set the main file path to `scripts/PredictionApp.py`
5. Deploy and share the public URL!

---

## Future Improvements

* Deploy on platforms like Heroku or Docker for more scalability
* Enable dynamic dataset updates or selections
* Add LSTM/GRU models for time-series forecasting
* Incorporate vaccine and demographic data for enhanced insights

---

## Contact

Feel free to reach out for collaboration or questions!

**Email:** \[huzaifa123ashrafi@gmail.com(mailto:huzaifa123ashrafi@gmail.com)]
**LinkedIn:** [www.linkedin.com/in/huzaifa-ashrafi](https://www.linkedin.com/in/huzaifa-ashrafi)

---
