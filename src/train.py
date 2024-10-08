import logging
#import mlflow
from sklearn.base import RegressorMixin
import pandas as pd

from src.config.logging_config import setup_logging
from src.config.model_config import ModelNameConfig
from src.data.data_loader import ingest_data, clean_data
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.linear_regression import LinearRegressionModel
from src.models.lgbm_model import LightGBMModel
from src.optimize.tuner import HyperparameterTuner
import logging


logger = setup_logging()

def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    try:
        model = None
        tuner = None

        if config.model_name == "lightgbm":
            #mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            #mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            #mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            #mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise e

if __name__ == "__main__":
    try:
        # Load and preprocess data
        raw_data = ingest_data("data/raw/landslides.csv")
        x_train, x_test, y_train, y_test = clean_data(raw_data)

        # Set up model configuration
        model_config = ModelNameConfig(model_name="randomforest", fine_tuning=True)

        # Train the model
        trained_model = train_model(x_train, x_test, y_train, y_test, model_config)

        # Here you can add code to save the model, evaluate it, etc.
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")