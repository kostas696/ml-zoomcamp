import os
import pandas as pd
import joblib
import logging
from dotenv import load_dotenv
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from preprocessing import create_preprocessing_pipeline
from preprocessing import replace_negatives_with_min


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

def main():
    try:
        # Load dataset path dynamically
        dataset_path = os.getenv("DATASET_PATH")
        if not dataset_path:
            logging.error("DATASET_PATH environment variable not set.")
            raise ValueError("DATASET_PATH environment variable not set.")

        logging.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # Drop 'pm2.5' and label encode the target variable
        if 'pm2.5' in df.columns:
            df = df.drop(columns=['pm2.5'])
            logging.info("Dropped 'pm2.5' column.")

        label_encoder_path = os.getenv("LABEL_ENCODER_PATH")
        if not label_encoder_path:
            logging.error("LABEL_ENCODER_PATH environment variable not set.")
            raise ValueError("LABEL_ENCODER_PATH environment variable not set.")

        le = joblib.load(label_encoder_path)
        logging.info(f"Label encoder loaded from {label_encoder_path}")
        df['air_quality_encoded'] = le.fit_transform(df['air_quality'])

        # Define feature set and target variable
        X = df.drop(columns=['air_quality', 'air_quality_encoded'])
        y = df['air_quality_encoded']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Split data into training and testing sets with shapes: {X_train.shape}, {X_test.shape}")

        # Create and apply preprocessing pipeline
        numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        preprocessor = create_preprocessing_pipeline(numerical_features)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        logging.info("Applied preprocessing pipeline to data.")

        # Save the preprocessor dynamically
        preprocessor_path = os.getenv("PREPROCESSOR_PATH")
        if not preprocessor_path:
            logging.error("PREPROCESSOR_PATH environment variable not set.")
            raise ValueError("PREPROCESSOR_PATH environment variable not set.")

        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Preprocessor saved at {preprocessor_path}")

        # Load best hyperparameters dynamically
        best_params_path = os.getenv("BEST_PARAMS_PATH")
        if not best_params_path:
            logging.error("BEST_PARAMS_PATH environment variable not set.")
            raise ValueError("BEST_PARAMS_PATH environment variable not set.")

        best_params = joblib.load(best_params_path)
        logging.info(f"Best hyperparameters loaded from {best_params_path}: {best_params}")

        # Train the model
        model = CatBoostClassifier(verbose=0, random_state=42, **best_params)
        model.fit(X_train_preprocessed, y_train)
        logging.info("Model trained successfully.")

        # Save the trained model dynamically
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            logging.error("MODEL_PATH environment variable not set.")
            raise ValueError("MODEL_PATH environment variable not set.")

        joblib.dump(model, model_path)
        logging.info(f"Trained model saved at {model_path}")

    except Exception as e:
        logging.exception("An error occurred during training.")

if __name__ == "__main__":
    main()
