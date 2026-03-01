from fastapi import FastAPI
import pandas as pd

from app.model_loader import load_model
from app.schema import CustomerData
from pipeline.preprocess import preprocess_data

app = FastAPI()
model_data = load_model()
model = model_data[0]
columns = model_data[1]


@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}


@app.post("/predict")
def predict(customer: CustomerData):
    
    # Convert to dataframe
    df = pd.DataFrame([customer.dict()])
    try:
        df = pd.DataFrame([customer.dict()])
    except Exception as e:
        return {"error": str(e)}

    
    # Apply preprocessing
    df_processed = preprocess_data(df)
    
    # Align columns
    df_processed = df_processed.reindex(columns=columns, fill_value=0)
    
    # Prediction
    prob = model.predict_proba(df_processed)[0][1]
    pred = model.predict(df_processed)[0]
    
    return {
        "churn_probability": float(prob),
        "churn_prediction": int(pred)
    }

@app.get("/health")
def health():
    return {"status": "ok"}