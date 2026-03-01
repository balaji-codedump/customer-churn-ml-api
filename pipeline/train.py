import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from preprocess import preprocess_data, split_features_target


# Load data
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Apply preprocessing
df_clean = preprocess_data(df)

# Split features and target
X, y = split_features_target(df_clean)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# Model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

# Train
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, probs))

model_data = {
    "model": model,
    "columns": X.columns.tolist()
}

joblib.dump(model_data, "models/xgboost_model.pkl")

print("Model and columns saved successfully")