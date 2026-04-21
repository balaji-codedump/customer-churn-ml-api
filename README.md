# Customer Churn Prediction ML API

An end-to-end Machine Learning API for predicting customer churn.  
Built using FastAPI and XGBoost, containerized with Docker, and deployed on AWS EC2.

---

## Features
- Data preprocessing pipeline  
- XGBoost classification model  
- REST API using FastAPI  
- Real-time predictions  
- Dockerized deployment  
- Cloud deployment on AWS EC2  

---

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Logistic Regression, RandomForest Classifier, LightGBM, FastAPI, Docker, AWS

---

## Live API
http://16.171.169.175:8000/docs

---

Input
{
  "gender": "Male",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.5
}

Output
{
  "churn_probability": 0.29,
  "churn_prediction": 0
}

Author

Balaji M
