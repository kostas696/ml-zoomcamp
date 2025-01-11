from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging
from dotenv import load_dotenv
import sys
import os

# Add the src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("predict.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load resources dynamically
try:
    preprocessor_path = os.getenv("PREPROCESSOR_PATH")
    model_path = os.getenv("MODEL_PATH")
    label_encoder_path = os.getenv("LABEL_ENCODER_PATH")

    if not all([preprocessor_path, model_path, label_encoder_path]):
        logging.error("One or more environment variables for file paths are missing.")
        raise ValueError("Environment variables for file paths are not properly configured.")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    logging.info("Preprocessor, model, and label encoder loaded successfully.")

except Exception as e:
    logging.exception("Failed to load preprocessor, model, or label encoder.")
    raise e

# Define input schema
class InputData(BaseModel):
    temperature: float
    humidity: float
    pm10: float
    no2: float
    so2: float
    co: float
    proximity_to_industrial_areas: float
    population_density: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Air Quality Prediction API!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        logging.info(f"Received input data: {input_df.to_dict(orient='records')}")

        # Apply preprocessing pipeline
        input_preprocessed = preprocessor.transform(input_df)
        logging.info("Input data preprocessed successfully.")

        # Make prediction
        prediction = model.predict(input_preprocessed)
        prediction_label = label_encoder.inverse_transform(prediction)[0]
        logging.info(f"Prediction: {prediction_label}")

        return {"prediction": prediction_label}

    except Exception as e:
        logging.exception("An error occurred during prediction.")
        raise HTTPException(status_code=500, detail="Prediction failed. Please check the server logs for details.")
