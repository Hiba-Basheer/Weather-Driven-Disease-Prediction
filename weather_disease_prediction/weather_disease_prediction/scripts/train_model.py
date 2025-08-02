import os
import sys
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Append the src directory to the Python path to enable relative imports
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))
sys.path.append(src_path)

from utils.preprocess import load_data, preprocess_data, get_features


def train_model():
    """
    Loads weather-related disease data, trains an XGBoost classifier,
    and saves the trained model along with the label encoder.
    """

    # Define file paths
    base_dir = os.path.join(
        "D:", "brototype", "week27", "weather driven disease prediction",
        "weather_disease_prediction", "weather_disease_prediction"
    )

    data_path = os.path.join(base_dir, "data", "raw", "Weather-related disease prediction.csv")
    model_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(model_dir, "trained_model.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    # Load and preprocess the data
    data = load_data(data_path)
    data, label_encoder = preprocess_data(data)
    X, y = get_features(data)

    # Remove any duplicate columns, if present
    X = X.loc[:, ~X.columns.duplicated()]

    # Split the dataset
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print a quick summary of training data
    print("🔍 Training data preview:")
    print(X_train.head(), "\n")
    print(f"🧾 Feature count: {X_train.shape[1]}")

    # Initialize and train the model
    model = XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save model and label encoder
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_encoder_path)

    print(f"\n✅ Model saved at: {model_path}")
    print(f"✅ Label encoder saved at: {label_encoder_path}")


if __name__ == "__main__":
    train_model()
