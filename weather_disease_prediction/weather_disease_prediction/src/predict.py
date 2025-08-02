import pandas as pd
from utils.preprocess import preprocess_data
from load_model import load_trained_model
from symptom_mapper import map_symptoms

def predict_disease(user_input, model_path, le_path, symptom_columns):
    """Predict disease based on user input."""
    model, le = load_trained_model(model_path, le_path)
    
    # Create input DataFrame
    input_data = {
        'Age': [user_input['age']],
        'Gender': [1 if user_input['gender'].lower() == 'male' else 0],
        'Temperature (C)': [user_input['temperature']],
        'Humidity': [user_input['humidity']],
        'Wind Speed (km/h)': [user_input['wind_speed']]
    }
    symptom_vector = map_symptoms(user_input, symptom_columns)
    for col, val in zip(symptom_columns, symptom_vector):
        input_data[col] = [val]
    
    input_df = pd.DataFrame(input_data)
    
    # Preprocess input (add lag features)
    input_df['Temperature_lag1'] = input_df['Temperature (C)'].shift(1).fillna(input_df['Temperature (C)'].mean())
    input_df['Humidity_lag1'] = input_df['Humidity'].shift(1).fillna(input_df['Humidity'].mean())
    input_df['WindSpeed_lag1'] = input_df['Wind Speed (km/h)'].shift(1).fillna(input_df['Wind Speed (km/h)'].mean())
    
    # Predict
    prediction = model.predict(input_df)[0]
    disease = le.inverse_transform([prediction])[0]
    return disease

