import joblib

def load_model():
    
    model_data = joblib.load("models/xgboost_model.pkl")
    
    model = model_data["model"]
    columns = model_data["columns"]
    
    return model, columns