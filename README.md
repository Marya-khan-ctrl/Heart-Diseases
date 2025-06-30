# Heart-Diseases

# Heart Disease Risk Prediction and Analysis

This project is a comprehensive Python-based implementation for heart disease risk prediction using machine learning models. It performs data preprocessing, exploratory data analysis, model training, evaluation, and includes a user input interface for live prediction.

## Project Objective

The goal is to predict the presence of heart disease in a patient using a set of medical attributes. The dataset is processed and analyzed, and two classification models (Logistic Regression and Decision Tree) are trained and evaluated. The project concludes with a console-based prediction system where users can input medical details and receive a prediction.

## Dataset

The dataset used is the Heart Disease dataset (commonly referred from UCI or Kaggle).  
Filename: `heart.csv`  
The target variable is `condition` (1: presence of heart disease, 0: absence).

### Features

- `age`: Age of the patient  
- `sex`: Sex (1 = male, 0 = female)  
- `cp`: Chest pain type (0-3)  
- `trestbps`: Resting blood pressure  
- `chol`: Serum cholesterol in mg/dl  
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg`: Resting electrocardiographic results (0-2)  
- `thalach`: Maximum heart rate achieved  
- `exang`: Exercise-induced angina (1 = yes, 0 = no)  
- `oldpeak`: ST depression induced by exercise  
- `slope`, `thal`: Categorical features  
- `condition`: Target variable (1 = disease, 0 = no disease)

## Dependencies

Install required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
