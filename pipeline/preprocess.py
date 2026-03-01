import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn")-> pd.DataFrame:

    df = df.copy()
    # target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # -------------------------
    # Binary Encoding
    # -------------------------
    binary_cols = [
        'gender', 'Partner', 'Dependents',
        'PhoneService', 'PaperlessBilling'
    ]

    existing_cols = [col for col in binary_cols if col in df.columns]

    df[existing_cols] = df[existing_cols].replace({
        'Yes': 1,
        'No': 0,
        'Male': 1,
        'Female': 0
    })

    # -------------------------
    # One Hot Encoding
    # -------------------------
    multi_cat_cols = [
        'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # -------------------------
    # TotalCharges Cleaning
    # -------------------------
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df = df.dropna(subset=['TotalCharges'])

    # -------------------------
    # Drop customerID
    # -------------------------
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # -------------------------
    # Boolean → Int
    # -------------------------
    bool_columns = df.select_dtypes(include='bool').columns
    df[bool_columns] = df[bool_columns].astype(int)

    # -------------------------
    # No Internet Service Collapse
    # -------------------------
    no_internet_cols = [
        'OnlineSecurity_No internet service',
        'OnlineBackup_No internet service',
        'DeviceProtection_No internet service',
        'TechSupport_No internet service',
        'StreamingTV_No internet service',
        'StreamingMovies_No internet service'
    ]

    existing_cols = [col for col in no_internet_cols if col in df.columns]

    if existing_cols:
        df['No_internet_service'] = df[existing_cols].any(axis=1).astype(int)
        df = df.drop(columns=existing_cols)

    # -------------------------
    # No Phone Service
    # -------------------------
    if 'MultipleLines_No phone service' in df.columns:
        df['No_phone_service'] = df['MultipleLines_No phone service']
        df = df.drop(columns=['MultipleLines_No phone service'])

    return df


def split_features_target(df: pd.DataFrame):
    """
    Splits features and target
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y