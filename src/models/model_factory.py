from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .linear_regression import LinearRegressionModel
from .lgbm_model import LightGBMModel
from src.config.model_config import MODEL_CONFIGS

def get_model(model_name):
    config = MODEL_CONFIGS.get(model_name, {})
    if model_name == "randomforest":
        return RandomForestModel(config)
    elif model_name == "xgboost":
        return XGBoostModel(config)
    elif model_name == "lightgbm":
        return LightGBMModel(config)
    elif model_name == "linear_regression":
        return LinearRegressionModel(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")