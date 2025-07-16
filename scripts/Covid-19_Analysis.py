# Data Handling & Visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud

#  ML Utilities
import joblib
from imblearn.over_sampling import SMOTE

#  Preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

#  Model Selection & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay
)

#  Feature Selection & Dimensionality Reduction
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

#  Statistical Tests
from scipy.stats import shapiro

#  Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Load the dataset
df = pd.read_csv("fadc_lab10.csv")

#list of symptoms
common_symptoms = [
    'fever', 'dry cough', 'fatigue', 'loss of taste', 'loss of or smell',
    'difficulty breathing', 'sore throat', 'headache', 'muscle aches',
    'chills', 'diarrhea', 'runny nose', 'nausea'
]

#Assign 6–7 random symptoms per row
df['Common_Symtoms'] = df.apply(lambda _: ', '.join(random.sample(common_symptoms, k=random.randint(6, 7))),axis=1)

#Split symptom strings into lists
print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates()
df['Common_Symtoms'] = df['Common_Symtoms'].str.lower().str.split(r',\s*')

#Classification of symptoms
for symptom in common_symptoms:
   df[symptom] = df['Common_Symtoms'].apply(lambda x: int(symptom in x))

# Combine all text entries
symptom_text = df['Common_Symtoms'].dropna().astype(str).str.lower().str.cat(sep=', ')

# Create WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(symptom_text)

# Display
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Symptoms")
plt.show()

#null values
print(df.isnull().sum())


# Checking for Duplicate Records
# 
# - The print statement shows the **number of duplicate rows** in the dataset.
# - Removing these duplicates ensures that **no data point is counted more than once**, which could otherwise distort the analysis.


#  outliers using IQR
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

    # cap them (Winsorization)
    df[col] = np.where(df[col] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR,
                       np.where(df[col] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR, df[col]))

# Data Preparation and Feature Engineering

# Drop 'iso_code' encode 'location'
if 'iso_code' in df.columns:
    df.drop(columns='iso_code', inplace=True)

# For visualization (keep all) & For modeling (classification)
if 'location' in df.columns:
    df_full = pd.get_dummies(df.copy(), columns=['location'], drop_first=False)
    df = pd.get_dummies(df, columns=['location'], drop_first=True)
else:
    df_full = df.copy()

# Drop 'Common_Symtoms' if it's redundant with symptom flags
df.drop(columns='Common_Symtoms', inplace=True)

# Date droped due to having error = could not convert string to float: '31/12/2019'
X = df.drop(columns=['total_deaths_per_million'])
y = df['total_deaths_per_million']

model = RandomForestRegressor()
selector = RFE(model, n_features_to_select=10)
selector = selector.fit(X, y)

selected_features = X.columns[selector.support_]
print("Selected features:", selected_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Creating Classification Target and Balancing the Dataset


# Classification target
df['high_death_risk'] = (df['total_deaths_per_million'] > df['total_deaths_per_million'].median()).astype(int)
print(df['high_death_risk'].value_counts())

# SMOTE for balancing
smote = SMOTE()
X_res, y_res = smote.fit_resample(df[selected_features], df['high_death_risk'])

# Dimensionality Reduction with PCA



# PCA to retain 95% of variance
pca = PCA(n_components=0.95)  # 0.95*100 =95%
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("Reduced shape after PCA:", X_pca.shape)


# Splitting PCA-Transformed Data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# Visualizing PCA Components

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - First Two Principal Components')
plt.colorbar(label='Target Variable')
plt.show()


# Checking data is Gaussian or Not Gaussian
# For target variable
sns.histplot(y, kde=True)
plt.title("Distribution of Target Variable")
plt.show()

# For a few features
for col in X.select_dtypes(include='number').columns[:3]:  # first 3 numerical features
    sns.histplot(X[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

stat, p = shapiro(y)
print(f"Shapiro-Wilk Test p-value: {p}")

if p > 0.05:
    print("Likely Gaussian (normal)")
else:
    print("Likely NOT Gaussian")

# Loop through all numerical features
for col in X.select_dtypes(include='number').columns:
    plt.figure(figsize=(6, 3))
    sns.histplot(X[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# Saving file after changes
df.to_csv("Cleaned Actual Data of Covid-19.csv", index=False)

#Regression Models

# 1.Linear Regression

# Justification  
# Linear Regression was used to predict continuous outcomes such as `total_deaths_per_million`. It is simple, interpretable, and effective when the relationship between the input features and the target is approximately linear. It helps in identifying general trends between numeric features.
# y = target variable
y = df['total_deaths_per_million']

# X = all other columns (entire dataset except the target)
X = df.drop('total_deaths_per_million', axis=1)
print(X.columns)
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# R-squared score (how well the model explains the data)
print("R² Score:", r2_score(y_test, y_pred))

# SSE: Sum of Squared Errors
sse = np.sum((y_test - y_pred) ** 2)
print("SSE (Sum of Squared Errors):", sse)

# MAE: Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("MAE (Mean Absolute Error):", mae)

# Mean Squared Error (how far predictions are from actual values)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Coefficients for each feature
coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients.sort_values(ascending=False))

# SSE: Sum of Squared Errors
sse = np.sum((y_test - y_pred) ** 2)
print("SSE (Sum of Squared Errors):", sse)

# MAE: Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("MAE (Mean Absolute Error):", mae)
plt.figure(figsize=(10, 6))

# Scatter plot: actual vs predicted values
plt.scatter(y_test, y_pred, color='teal', edgecolor='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')

plt.xlabel('Actual Total Deaths per Million')
plt.ylabel('Predicted Total Deaths per Million')
plt.title('Actual vs Predicted Values (Linear Regression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# R-squared
r2 = r2_score(y_test, y_pred)

# MSE
mse = mean_squared_error(y_test, y_pred)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# SSE (Sum of Squared Errors)
sse = np.sum((y_test - y_pred) ** 2)

# Print all metrics
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"SSE (Sum of Squared Errors): {sse:.4f}")
y_pred = model.predict(X_test)


#Polynomial Regression

# Justification 
# Polynomial Regression (Degree 2) was used to capture **non-linear** relationships in the data. The model adds \( x^2 \) features to account for curvature in data trends. It improved performance when Linear Regression failed to fit complex data patterns adequately.
# Choose degree of the polynomial
degree = 2

poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)  # Transforms original features into polynomial features

# Split the polynomial features
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Fit model
poly_model = LinearRegression()
poly_model.fit(X_train, y_train)

# Predict
y_pred = poly_model.predict(X_test)

# Evaluation Metrics
print("Polynomial Regression (Degree = {})".format(degree))
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("SSE:", np.sum((y_test - y_pred) ** 2))

plt.figure(figsize=(10, 6))

# Actual vs predicted plot
plt.scatter(y_test, y_pred, color='orange', edgecolor='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Ideal Prediction')

plt.title(f'Actual vs Predicted (Polynomial Regression)')
plt.xlabel('Actual Total Deaths per Million')
plt.ylabel('Predicted Total Deaths per Million')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# # Classfication Models
# Target
y = df['high_death_risk']
# Drop the target from X
X = df.drop(columns=['high_death_risk'])
print(X.columns)
print(y.head())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 1st Model:Logistic Regression
# Logistic Regression was used for binary classification tasks, such as predicting `high_death_risk (0 or 1)`. It provides probability estimates and is robust to correlated features. It’s particularly useful when the relationship between features and the binary target is linear in the log-odds.

# Create a pipeline that imputes missing values and then fits logistic regression
model = make_pipeline(SimpleImputer(strategy='most_frequent'), LogisticRegression(max_iter=1000))

model.fit(X_train, y_train)

# Predict and print accuracy
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# OR print them all together
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Other Classification Models

# Dictionary of models
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Loop through each model
for name, model in models.items():
    print(f"\n Model: {name}")

    # Create pipeline with imputer and classifier
    pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), model)

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1 Score : {f1:.4f}")

    # Display full classification report
    print(f"\n Classification Report:\n{classification_report(y_test, y_pred)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# # **ROC-AUC Curve for both train and test dataset**

# # KNN(K-Neighbors Classifier)

# Pipeline
knn_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), KNeighborsClassifier())
knn_pipeline.fit(X_train, y_train)

# Predict probabilities
y_train_prob = knn_pipeline.predict_proba(X_train)[:, 1]
y_test_prob = knn_pipeline.predict_proba(X_test)[:, 1]

# ROC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# AUC
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# Plot
plt.figure()
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Decision Tree

# Pipeline
dt_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), DecisionTreeClassifier(random_state=42))
dt_pipeline.fit(X_train, y_train)

# Predict probabilities
y_train_prob = dt_pipeline.predict_proba(X_train)[:, 1]
y_test_prob = dt_pipeline.predict_proba(X_test)[:, 1]

# ROC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# AUC
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# Plot
plt.figure()
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# # Naive Bayes

# Pipeline
nb_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), GaussianNB())
nb_pipeline.fit(X_train, y_train)

# Predict probabilities
y_train_prob = nb_pipeline.predict_proba(X_train)[:, 1]
y_test_prob = nb_pipeline.predict_proba(X_test)[:, 1]

# ROC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# AUC
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# Plot
plt.figure()
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()


# # Logistic Regression

# Predict probabilities (for class 1)
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for train
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and AUC for test
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
auc_test = auc(fpr_test, tpr_test)

# Plot ROC curve
plt.figure()
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}')
plt.plot([0, 1], [0, 1], 'k--')

plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

features = [
    'fever', 'dry cough', 'fatigue', 'loss of taste', 'loss of or smell',
    'difficulty breathing', 'sore throat', 'headache', 'muscle aches',
    'chills', 'diarrhea', 'runny nose', 'nausea'
]

# Select target 
X = df[features]
y = df['total_deaths_per_million']

# 1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
joblib.dump(linear_model, 'linear_model.pkl')

# 2. Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_poly_train)
joblib.dump(poly_model, 'poly_model.pkl')

# 3. Random Forest Regressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X, y)
joblib.dump(rf_regressor, 'rf_regressor.pkl')

# Select target
y = df['high_death_risk']

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, 'rf_classifier.pkl')

# 5. Logistic Regression (with pipeline)
logistic_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), LogisticRegression(max_iter=1000))
logistic_pipeline.fit(X_train, y_train)
joblib.dump(logistic_pipeline, 'logistic_model.pkl')

# 6. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
joblib.dump(knn, 'knn_model.pkl')

# 7. Decision Tree Classifier (with pipeline)
dt_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), DecisionTreeClassifier(random_state=42))
dt_pipeline.fit(X_train, y_train)
joblib.dump(dt_pipeline, 'decision_tree_model.pkl')

# 8. Naive Bayes (Bernoulli)
nb = BernoulliNB()
nb.fit(X_train, y_train)
joblib.dump(nb, 'naive_bayes_model.pkl')

# Sample user input (you can use input() for interactive mode)
user_input = {
    'fever': 1,
    'dry cough': 1,
    'fatigue': 0,
    'loss of taste': 1,
    'loss of or smell': 0,
    'difficulty breathing': 0,
    'sore throat': 1,
    'headache': 0,
    'muscle aches': 0,
    'chills': 0,
    'diarrhea': 0,
    'runny nose': 0,
    'nausea': 0
}


# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]

# Output result
if prediction == 1:
    print("High death risk – needs urgent medical care")
else:
    print("Low death risk – symptoms manageable")

# Step 7: Load model
# Choose one: 'knn_model.pkl' or 'logistic_model.pkl'
model = joblib.load('logistic_model.pkl')  # or 'logistic_model.pkl'

# Sample user input (you can use input() for interactive mode)
user_input = {
    'fever': 1,
    'dry cough': 1,
    'fatigue': 0,
    'loss of taste': 1,
    'loss of or smell': 0,
    'difficulty breathing': 0,
    'sore throat': 1,
    'headache': 0,
    'muscle aches': 0,
    'chills': 0,
    'diarrhea': 0,
    'runny nose': 0,
    'nausea': 0
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]

# Output result
if prediction == 1:
    print("High death risk – needs urgent medical care")
else:
    print("Low death risk – symptoms manageable")

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Plot feature importance
importances = rf_model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Which Symptoms Contribute Most to High Death Risk")
plt.gca().invert_yaxis()
plt.show()

# Sample user input (you can use input() for interactive mode)
user_input = {
    'fever': 0,
    'dry cough': 1,
    'fatigue': 0,
    'loss of taste': 1,
    'loss of or smell': 0,
    'difficulty breathing': 0,
    'sore throat': 1,
    'headache': 0,
    'muscle aches': 1,
    'chills': 0,
    'diarrhea': 0,
    'runny nose': 1,
    'nausea': 1
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]

# Output result
if prediction == 1:
    print("High death risk – needs urgent medical care")
else:
    print("Low death risk – symptoms manageable")

# Group by 'death' (target column), then count how many fever=1 and fever=0
fever_counts = df.groupby(['high_death_risk', 'chills']).size().unstack(fill_value=0)

# Rename the index/columns for clarity
fever_counts.index.name = 'Death Risk'
fever_counts.columns = ['fever=0', 'fever=1']

print("Fever Counts by Death Risk Class:")
print(fever_counts)