import logging
import mlflow
from sklearn.base import RegressorMixin
import pandas as pd
import numpy as np
from src.config.logging_config import setup_logging
from src.config.model_config import ModelNameConfig
from src.data.data_loader import ingest_data, clean_data
from src.models.model_factory import get_model
from src.optimize.tuner import HyperparameterTuner
import logging
from src.config.model_config import ModelNameConfig, MODEL_CONFIGS

logger = setup_logging()

def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    try:
        model_config = MODEL_CONFIGS[config.model_name]
        model = get_model(config.model_name)
        
        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
        elif config.model_name in ["randomforest", "linear_regression"]:
            mlflow.sklearn.autolog()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            model_config.update(best_params)
            trained_model = model.train(x_train, y_train)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise e

if __name__ == "__main__":
    try:
        # Load and preprocess data
        raw_data = ingest_data("/home/joydip/Documents/Devlopment/LandslidePredictor---End-to-End-MLOps-Solution-for-Landslide-Risk-Prediction/data/synthetic_landslide_data_2.csv")
        x_train, x_test, y_train, y_test = clean_data(raw_data)

        # Set up model configuration
        model_config = ModelNameConfig(model_name="xgboost", fine_tuning=True)

        # Train the model
        trained_model = train_model(x_train, x_test, y_train, y_test, model_config)

        # Here you can add code to save the model, evaluate it, etc.
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")