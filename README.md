# Customer Churn Prediction ML API

An end-to-end Machine Learning API for predicting customer churn.  
Built using FastAPI and XGBoost, with Docker for deployment.

## Features
- Data preprocessing pipeline
- XGBoost classification model
- REST API with FastAPI
- Real-time predictions
- Dockerized deployment

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, FastAPI, Docker

## Project Structure
app/ # FastAPI app
pipeline/ # Preprocessing code
model/ # Trained model
notebook/ # EDA and training
Dockerfile
requirements.txt

## Run Locally
pip install -r requirements.txt
uvicorn app.main:app --reload

Open:  
http://localhost:8000/docs

## Run with Docker
docker build -t churn-api .
docker run -p 8000:8000 churn-api

Open:  
http://localhost:8000/docs

## API Example

### Input
json
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

### Output
{
  "churn_probability": 0.32,
  "churn_prediction": 0
}


### Author
Balaji M
