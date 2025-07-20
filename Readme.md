![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)
![License](https://img.shields.io/github/license/HuzaifaAshrafi1/COVID19-ML-Prediction)

#  COVID-19 Death Risk Predictor

A comprehensive machine learning project that analyzes COVID-19 data, builds predictive models, and deploys a Streamlit web app to estimate death risk based on user symptoms.

---

## Overview

This project leverages data analysis, visualization, and machine learning (regression + classification) to predict COVID-19 death risks per country and individual input.

Key features:
- Symptom-based prediction
- Time & location-based feature engineering
- Multiple regression and classification models
- Interactive Streamlit app for live predictions
- PCA and outlier handling for improved accuracy

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
│   ├── linear\_regression\_model.pkl
│   ├── polynomial\_regression\_model.pkl
│   ├── random\_forest\_model.pkl
│   ├── decision\_tree\_model.pkl
│   ├── knn\_model.pkl
│   ├── naive\_bayes\_model.pkl
│   └── logistic\_regression\_model.pkl
│
├── scripts/
│   ├── PredictionApp.py                    # Streamlit Web App
│   └── COVID 19 Analysis,ML & Models.ipynb # Full Data Analysis & Modeling
│
├── visuals/
│   └── \*.jpg (Plots, PCA, Boxplots, Confusion Matrix, etc.)
│
├── requirements.txt
└── README.md

````

---

## Machine Learning Models

### Regression
-  Linear Regression
-  Polynomial Regression (degree=2)

### Classification
-  Logistic Regression
-  Decision Tree Classifier
-  Random Forest Classifier
-  K-Nearest Neighbors (KNN)
-  Naive Bayes

---

## Visualizations

- Word Cloud (Symptom Frequency)
- Distributions (Histograms, KDEs)
- PCA Plot (2D projection)
- Boxplots (Outliers)
- Confusion Matrices
- ROC-AUC Curves

---

## Web Application (Streamlit)

### Features:
- Choose between Regression or Classification
- Select a prediction model
- Input symptoms manually (1 = Yes, 0 = No)
- For regression: input total cases per million
- Get real-time predictions with risk interpretation

### 💻 Run the app:

```bash
python -m streamlit run scripts/PredictionApp.py
````

---

## Installation

Make sure Python 3.7+ is installed. Then:

```bash
# Clone the repository
git clone https://github.com/HuzaifaAshrafi1/COVID19-ML-Prediction.git

# Navigate to the project directory
cd COVID19-ML-Prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # On Windows use: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run scripts/PredictionApp.py

```

---

## Requirements

All required libraries are in `requirements.txt`:

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

To install:

```bash
pip install -r requirements.txt
```

---

## Data Cleaning Includes:

* Handling missing values
* Encoding categorical variables
* Creating time-based features (year, month, day)
* Treating outliers using IQR & Winsorization
* Symptom flagging & feature creation

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

## Future Improvements

* Deploy app on **Streamlit Cloud**, **Heroku**, or **Docker**
* Enable dynamic dataset selection or updates
* Use LSTM/GRU for time-series modeling
* Add vaccine and demographic data for deeper insights

## Contact

Feel free to reach out if you'd like to contribute, collaborate, or need assistance!

Email: [huzaifa123ashrafi@gmail.com]
Portfolio: [www.linkedin.com/in/huzaifa-ashrafi]
