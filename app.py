# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             confusion_matrix, roc_curve, 
                             classification_report)

# Load the dataset
df = pd.read_csv("heart.csv")

# Data Cleaning
print("Initial Data Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# EDA (optional display)
plt.figure(figsize=(12, 8))

# 1. Target Variable Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='condition', data=df)
plt.title("Heart Disease Distribution")

# 2. Age Distribution
plt.subplot(2, 2, 2)
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")

# 3. Cholesterol vs Heart Disease
plt.subplot(2, 2, 3)
sns.boxplot(x='condition', y='chol', data=df)
plt.title("Cholesterol by Heart Disease Status")

# 4. Max Heart Rate vs Heart Disease
plt.subplot(2, 2, 4)
sns.boxplot(x='condition', y='thalach', data=df)
plt.title("Max Heart Rate by Heart Disease Status")

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Feature Engineering
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

# Split into features and target
X = df.drop('condition', axis=1)
y = df['condition']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test_scaled)
y_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

y_pred_dt = dt.predict(X_test_scaled)
y_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_log))
print("\n--- Decision Tree ---")
print(classification_report(y_test, y_pred_dt))

print("\nModel Comparison:")
print(f"Logistic Regression - Accuracy: {accuracy_score(y_test, y_pred_log):.2f}, ROC-AUC: {roc_auc_score(y_test, y_proba_log):.2f}")
print(f"Decision Tree - Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}, ROC-AUC: {roc_auc_score(y_test, y_proba_dt):.2f}")

# Confusion Matrices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Greens')
plt.title("Decision Tree Confusion Matrix")

plt.tight_layout()
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)

plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba_log):.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_score(y_test, y_proba_dt):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# Feature Importance - Logistic Regression
plt.figure(figsize=(10, 6))
log_coef = pd.DataFrame(log_reg.coef_[0], index=X.columns, columns=['Coefficient']).sort_values(by='Coefficient')
sns.barplot(x='Coefficient', y=log_coef.index, data=log_coef)
plt.title("Logistic Regression Feature Importance")
plt.tight_layout()
plt.show()

# Feature Importance - Decision Tree
plt.figure(figsize=(10, 6))
dt_importance = pd.DataFrame(dt.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance')
sns.barplot(x='Importance', y=dt_importance.index, data=dt_importance)
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()

# --------------------
# USER INPUT SECTION
# --------------------

print("\n--- Heart Disease Risk Predictor ---")

def get_input(prompt, cast_type=float):
    while True:
        try:
            return cast_type(input(prompt + ": "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")

user_data = {
    "age": get_input("Enter age", int),
    "sex": get_input("Sex (1 = male, 0 = female)", int),
    "trestbps": get_input("Resting blood pressure"),
    "chol": get_input("Cholesterol level (mg/dl)"),
    "fbs": get_input("Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)", int),
    "thalach": get_input("Maximum heart rate achieved"),
    "exang": get_input("Exercise-induced angina (1 = yes, 0 = no)", int),
    "oldpeak": get_input("ST depression (oldpeak)"),
}

# One-hot encoding inputs
for i in range(1, 4):
    user_data[f'cp_{i}'] = 0
cp_val = int(get_input("Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)", int))
if cp_val in [1, 2, 3]:
    user_data[f'cp_{cp_val}'] = 1

for i in range(1, 3):
    user_data[f'restecg_{i}'] = 0
rest_val = int(get_input("Resting ECG (0: normal, 1: ST abnormality, 2: LV hypertrophy)", int))
if rest_val in [1, 2]:
    user_data[f'restecg_{rest_val}'] = 1

for i in range(1, 3):
    user_data[f'slope_{i}'] = 0
slope_val = int(get_input("Slope of ST segment (0: up, 1: flat, 2: down)", int))
if slope_val in [1, 2]:
    user_data[f'slope_{slope_val}'] = 1

for i in range(1, 3):
    user_data[f'thal_{i}'] = 0
thal_val = int(get_input("Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)", int))
if thal_val in [1, 2]:
    user_data[f'thal_{thal_val}'] = 1

# Build DataFrame
user_df = pd.DataFrame([user_data])

# Ensure all required columns are present
for col in X.columns:
    if col not in user_df.columns:
        user_df[col] = 0

user_df = user_df[X.columns]  # Ensure correct column order

# Scale and predict
user_scaled = scaler.transform(user_df)
user_pred = log_reg.predict(user_scaled)[0]
user_proba = log_reg.predict_proba(user_scaled)[0][1]

# Output prediction
print("\n--- Prediction Result ---")
print("Heart Disease Risk:", "HIGH" if user_pred == 1 else "LOW")
print(f"Probability of Heart Disease: {user_proba:.2f}")
