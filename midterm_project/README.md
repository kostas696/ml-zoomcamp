# Midterm Project

## Gym Members Calories Prediction with CatBoost

---

## Overview

This project predicts the number of calories burned by gym members during exercise sessions based on health and activity features. The model was trained using CatBoost, achieving high accuracy, and deployed as a web service via FastAPI. The service is containerized using Docker for easy deployment.

---

## Problem Description

Accurate calorie prediction supports:
- Personalized fitness planning.
- Progress tracking towards health goals.
- Insights for gym members to optimize their workouts.

The objective is to build and deploy a regression model for real-time calorie prediction.

---

## Data Overview

The dataset includes:
- **Numerical Features**:
  - Session Duration
  - Workout Intensity
  - Fat Percentage
  - BMI
  - Heart Rate Difference
  - Age
  - Height
  - Resting BPM
  - Water Intake
  - Workout Frequency
- **Target Variable**: `calories_burned`

Preprocessing steps include feature engineering, encoding categorical features, and feature selection.

---

## Workflow and Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed distributions and correlations.
- Addressed missing and redundant data.

### 2. Feature Engineering
- Engineered derived features:
  - **Workout Intensity** = Calories Burned / Session Duration
  - **Heart Rate Difference** = Max BPM - Resting BPM
- Applied feature selection using RFE and SHAP values.

### 3. Model Training
- Experimented with models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - CatBoost
- Tuned CatBoost hyperparameters with **Optuna**.
- Best RMSE on test set: **8.13**.

### 4. Model Deployment
- Built a FastAPI service for prediction.
- Containerized the service using Docker.

---

## Repository Structure

```plaintext
├── README.md                 # Project description and instructions
├── data/                     # Raw and processed datasets
├── notebooks/                # EDA and experimentation
│   └── notebook.ipynb
├── src/                      # Source code
│   ├── train.py              # Model training script
│   ├── predict.py            # Prediction service script
├── models/                   # Saved models and artifacts
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker container configuration
├── environment.yml           # Conda environment configuration
├── deployment_screenshots/   # Deployment screenshots
```

## Reproducibility
1. Clone the Repository
git clone https://github.com/kostas696/ml-zoomcamp.git
cd ml-zoomcamp/midterm_project

2. Set Up the Environment
Ensure conda is installed on your system. Create and activate the environment using the provided environment.yml file:
conda env create -f environment.yml
conda activate gym-calories-env

3. Download the Dataset
Download the dataset from Kaggle and place it in the data folder.

4. Install Additional Dependencies
If required, install additional dependencies:
pip install -r requirements.txt

## Model Training and Deployment Workflow
Training the Model
Train the model and save artifacts:

python src/train.py

Running the Prediction Service

Start the FastAPI Service

Run the FastAPI service locally:

uvicorn src.predict:app --host 0.0.0.0 --port 8000 --reload

## Testing the Service
Test the Service Using curl
Send a POST request to test the prediction service:

curl -X 'POST' \
     'http://127.0.0.1:8000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '[
         {
             "session_duration": 1.5,
             "workout_intensity": 750.0,
             "fat_percentage": 20.0,
             "bmi": 25.0,
             "heart_rate_difference": 50,
             "age": 30,
             "height": 1.75,
             "resting_bpm": 60,
             "water_intake": 2.5,
             "workout_frequency": 3
         },
         {
             "session_duration": 2.0,
             "workout_intensity": 800.0,
             "fat_percentage": 22.0,
             "bmi": 24.0,
             "heart_rate_difference": 60,
             "age": 28,
             "height": 1.80,
             "resting_bpm": 58,
             "water_intake": 3.0,
             "workout_frequency": 4
         }
     ]'


### Example Response

{
  "predictions": [
    1120.31,
    1469.27
  ]
}

## Deployment

### Containerization with Docker
Build the Docker Image

docker build -t gym-calories-service .

Start the Docker Container
Expose the application on port 8000:

docker run -p 8000:8000 gym-calories-service

## Results
Best Model: CatBoost
Test RMSE: 8.13

## Key Features:
*Session Duration* - 
*Workout Intensity*

## Limitations

Cloud deployment was not implemented but could be seamlessly integrated using platforms like AWS, GCP, or Azure.
The model retraining pipeline is currently manual and could benefit from automation through CI/CD workflows.

## Conclusion
This project highlights:

- Comprehensive exploratory data analysis (EDA) and feature engineering.
- Effective model training and hyperparameter tuning.
- A practical deployment of the model as a containerized web service for real-time predictions.
