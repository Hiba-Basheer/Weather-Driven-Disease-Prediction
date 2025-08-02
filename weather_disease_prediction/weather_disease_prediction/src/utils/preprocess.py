import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Clean and preprocess the data.

    - Drops missing values
    - Encodes categorical variables
    - Adds lag features

    Args:
        data (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Preprocessed data.
        LabelEncoder: Fitted label encoder (for decoding later).
    """
    # Drop missing rows
    data = data.dropna()

    # Encode 'Gender' and 'prognosis' columns
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"])
    data["prognosis"] = le.fit_transform(data["prognosis"])

    # Create lag features for weather-related attributes
    data["Temperature_lag1"] = data["Temperature (C)"].shift(1).fillna(data["Temperature (C)"].mean())
    data["Humidity_lag1"] = data["Humidity"].shift(1).fillna(data["Humidity"].mean())
    data["WindSpeed_lag1"] = data["Wind Speed (km/h)"].shift(1).fillna(data["Wind Speed (km/h)"].mean())

    return data, le


def get_features(data):
    """
    Extract features and target from the dataset.

    Args:
        data (pd.DataFrame): Preprocessed dataset.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
    """
    # Target variable
    y = data["prognosis"]

    # Feature matrix (remove target column and duplicates)
    X = data.drop(columns=["prognosis"])
    X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicate columns

    return X, y
