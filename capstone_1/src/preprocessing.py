import pandas as pd
import os
import joblib
import logging
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Custom transformation to replace negative values
def replace_negatives_with_min(X):
    X = X.copy()
    for col in X.select_dtypes(include=["float64", "int64"]).columns:
        if (X[col] < 0).any():
            min_positive = X[X[col] > 0][col].min()
            if pd.isna(min_positive):
                logging.error(f"No positive values found for column '{col}' to replace negatives.")
                raise ValueError(f"No positive values found for column '{col}' to replace negatives.")
            logging.info(f"Replacing negative values in column '{col}' with {min_positive}")
            X[col] = X[col].apply(lambda x: x if x >= 0 else min_positive)
    return X

# Preprocessing pipeline
def create_preprocessing_pipeline(numerical_features):
    logging.info("Creating preprocessing pipeline...")
    numerical_transformer = Pipeline(steps=[
        ('negative_value_replacement', FunctionTransformer(replace_negatives_with_min)),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

if __name__ == "__main__":
    try:
        # Load dataset dynamically from environment variables
        dataset_path = os.getenv("DATASET_PATH")
        if not dataset_path:
            logging.error("DATASET_PATH environment variable not set.")
            raise ValueError("DATASET_PATH environment variable not set.")
        
        logging.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # Identify numerical features
        numerical_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        logging.info(f"Numerical features identified: {numerical_features}")

        # Create preprocessing pipeline
        pipeline = create_preprocessing_pipeline(numerical_features)

        # Save pipeline dynamically
        preprocessor_path = os.getenv("PREPROCESSOR_PATH")
        if not preprocessor_path:
            logging.error("PREPROCESSOR_PATH environment variable not set.")
            raise ValueError("PREPROCESSOR_PATH environment variable not set.")
        
        joblib.dump(pipeline, preprocessor_path)
        logging.info(f"Preprocessing pipeline saved at {preprocessor_path}")

    except Exception as e:
        logging.exception("An error occurred during preprocessing.")
