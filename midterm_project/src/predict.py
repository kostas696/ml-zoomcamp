import warnings
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Suppress specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Paths
model_path = r"C:\Users\User\ml-zoomcamp\midterm_project\models\catboost_final_model.pkl"
scaler_path = r"C:\Users\User\ml-zoomcamp\midterm_project\models\scaler.pkl"

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for input data
class InputData(BaseModel):
    session_duration: float
    workout_intensity: float
    fat_percentage: float
    bmi: float
    heart_rate_difference: float
    age: float
    height: float
    resting_bpm: float
    water_intake: float
    workout_frequency: float

# Endpoint to check service status
@app.get("/")
def read_root():
    return {"message": "Model is ready to serve predictions"}

# Prediction endpoint
@app.post("/predict")
def predict(data: List[InputData]):
    # Convert input data to a list of lists
    input_features = [[
        d.session_duration,
        d.workout_intensity,
        d.fat_percentage,
        d.bmi,
        d.heart_rate_difference,
        d.age,
        d.height,
        d.resting_bpm,
        d.water_intake,
        d.workout_frequency
    ] for d in data]

    # Scale the input features
    scaled_features = scaler.transform(input_features)

    # Make predictions
    predictions = model.predict(scaled_features)
    return {"predictions": predictions.tolist()}