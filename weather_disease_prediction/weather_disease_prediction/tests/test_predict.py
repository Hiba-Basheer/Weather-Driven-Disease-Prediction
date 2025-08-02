# tests/test_predict.py

import os
import sys
import pytest

# Add the src directory to sys.path
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(
    os.path.join(current_dir, "..", "..", "weather_disease_prediction", "src")
)
sys.path.append(src_dir)

from predict import predict_disease


def test_predict_disease():
    """
    Unit test for predict_disease function.
    Checks if the return type is a string.
    """
    symptom_columns = [
        "nausea", "joint_pain", "abdominal_pain", "high_fever", "chills", "fatigue",
        "runny_nose", "pain_behind_the_eyes", "dizziness", "headache", "chest_pain",
        "vomiting", "cough", "shivering", "asthma_history", "high_cholesterol",
        "diabetes", "obesity", "hiv_aids", "nasal_polyps", "asthma",
        "high_blood_pressure", "severe_headache", "weakness", "trouble_seeing",
        "fever", "body_aches", "sore_throat", "sneezing", "diarrhea",
        "rapid_breathing", "rapid_heart_rate", "pain_behind_eyes", "swollen_glands",
        "rashes", "sinus_headache", "facial_pain", "shortness_of_breath",
        "reduced_smell_and_taste", "skin_irritation", "itchiness", "throbbing_headache",
        "confusion", "back_pain", "knee_ache"
    ]

    user_input = {
        "age": 30,
        "gender": "male",
        "temperature": 25.0,
        "humidity": 0.7,
        "wind_speed": 10.0,
        "symptoms": ["nausea", "high_fever"]
    }

    disease = predict_disease(
        user_input,
        model_path="../models/trained_model.pkl",
        encoder_path="../models/label_encoder.pkl",
        symptom_columns=symptom_columns
    )

    assert isinstance(disease, str), "Prediction should return a string"
