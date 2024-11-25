import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import json

def train_model():
    try:
        print("Starting model training...")
        processed_data_path = r"C:/Users/User/ml-zoomcamp/midterm_project/data/processed/encoded_data.csv"
        model_output_path = r"C:/Users/User/ml-zoomcamp/midterm_project/models/catboost_final_model.pkl"
        hyperparams_path = r"C:/Users/User/ml-zoomcamp/midterm_project/models/best_hyperparams.json"
        selected_features_path = r"C:/Users/User/ml-zoomcamp/midterm_project/models/selected_features.json"

        # Load selected features
        with open(selected_features_path, 'r') as file:
            selected_features = json.load(file)

        # Load the processed dataset
        df_encoded = pd.read_csv(processed_data_path)

        # Define target and features
        X = df_encoded[selected_features]
        y = df_encoded['calories_burned']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler
        scaler_path = r"C:/Users/User/ml-zoomcamp/midterm_project/models/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

        # Load best hyperparameters
        with open(hyperparams_path, 'r') as file:
            best_hyperparams = json.load(file)

        # Train the final model with the best hyperparameters
        final_model = CatBoostRegressor(
            iterations=best_hyperparams['iterations'],
            learning_rate=best_hyperparams['learning_rate'],
            depth=best_hyperparams['depth'],
            l2_leaf_reg=best_hyperparams['l2_leaf_reg'],
            subsample=best_hyperparams['subsample'],
            verbose=0,
            random_state=42
        )

        # Combine training data and train the final model
        final_model.fit(X_train_scaled, y_train)

        # Save the final model
        joblib.dump(final_model, model_output_path)
        print(f"Final model saved to {model_output_path}")

        print("Model training completed")
        
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise

if __name__ == "__main__":
    print("Script started")
    print("Script is running")
    train_model()