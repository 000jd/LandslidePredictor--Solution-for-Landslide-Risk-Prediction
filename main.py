import argparse
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime

from src.config.logging_config import setup_logging
from src.config.model_config import ModelNameConfig
from src.data.data_loader import ingest_data, clean_data
from src.models.model_factory import get_model
from src.train import train_model
from src.evaluation import MSE, R2Score, RMSE

logger = setup_logging()

def save_model(model, model_name):
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    import joblib
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def check_data(X, y, dataset_name):
    logger.info(f"Checking {dataset_name} dataset...")
    
    # Check for NaN values
    if X.isnull().any().any():
        logger.warning(f"NaN values found in {dataset_name} features")
    if isinstance(y, pd.Series) and y.isnull().any():
        logger.warning(f"NaN values found in {dataset_name} target")
    
    # Check numeric columns for non-finite values
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    if not np.isfinite(X[numeric_columns]).all().all():
        logger.warning(f"Non-finite values found in {dataset_name} numeric features")
    
    # Check categorical columns
    categorical_columns = X.select_dtypes(include=['category']).columns
    for col in categorical_columns:
        if X[col].isnull().any():
            logger.warning(f"NaN values found in categorical column '{col}' in {dataset_name} features")
    
    # Check target variable
    if isinstance(y, pd.Series):
        if y.dtype == 'category':
            if y.isnull().any():
                logger.warning(f"NaN values found in {dataset_name} categorical target")
        elif not np.isfinite(y).all():
            logger.warning(f"Non-finite values found in {dataset_name} numeric target")

def main(model_name, fine_tuning):
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        raw_data = ingest_data("/home/joydip/Documents/Devlopment/LandslidePredictor---End-to-End-MLOps-Solution-for-Landslide-Risk-Prediction/data/synthetic_landslide_data_2.csv")
        x_train, x_test, y_train, y_test = clean_data(raw_data)

        # Check data for issues
        check_data(x_train, y_train, "training")
        check_data(x_test, y_test, "testing")

        # Set up model configuration
        logger.info(f"Setting up {model_name} model with fine_tuning={fine_tuning}")
        model_config = ModelNameConfig(model_name=model_name, fine_tuning=fine_tuning)

        # Train the model
        logger.info("Training the model...")
        trained_model = train_model(x_train, x_test, y_train, y_test, model_config)

        # Save the model
        save_model(trained_model, model_name)

        # Evaluate the model
        logger.info("Evaluating the model...")
        y_pred = trained_model.predict(x_test)
        mse = MSE().calculate_score(y_test, y_pred)
        r2 = R2Score().calculate_score(y_test, y_pred)
        rmse = RMSE().calculate_score(y_test, y_pred)

        logger.info(f"Model evaluation results:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")

        logger.info("Model training and evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a landslide prediction model")
    parser.add_argument("--model", type=str, default="randomforest", choices=["randomforest", "xgboost", "lightgbm", "linear_regression"], help="Model to train")
    parser.add_argument("--train", action="store_true", help="Enable with hyperparameter fine-tuning")
    args = parser.parse_args()

    main(args.model, args.train)