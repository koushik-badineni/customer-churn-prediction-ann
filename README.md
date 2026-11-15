# Telecom Customer Churn Prediction Using ANN

## Project Overview

An end-to-end Machine Learning and Deep Learning project that predicts whether a telecom customer is likely to leave the company using an Artificial Neural Network (ANN).

The application is developed using TensorFlow and Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Artificial Neural Network (ANN)
- Streamlit
- Matplotlib

## Dataset

IBM Telco Customer Churn Dataset.

Dataset source:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Project Workflow

1. Data collection
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Train-test splitting
6. Preprocessing pipeline
7. ANN model development
8. Model evaluation
9. Streamlit application development

## Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 76.37% |
| Churn Precision | 53.98% |
| Churn Recall | 74.33% |
| Churn F1-score | 62.54% |
| ROC-AUC | 84.12% |

## Streamlit Application Features

- Customer information input form
- Churn probability prediction
- Customer risk classification
- Business recommendations
- Prediction history
- Analytics dashboard
- CSV export

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Author

Manikanta Koushik