import pandas as pd
from pipeline.preprocess import preprocess_data

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df_clean = preprocess_data(df)

print(df_clean.shape)
print(df_clean.head())