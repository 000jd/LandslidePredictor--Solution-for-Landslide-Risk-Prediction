from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .linear_regression import LinearRegressionModel
from .lgbm_model import LightGBMModel

def get_model(model_name):
    if model_name == "randomforest":
        return RandomForestModel()
    elif model_name == "xgboost":
        return XGBoostModel()
    elif model_name == "lightgbm":
        return LightGBMModel()
    elif model_name == "linear_regression":
        return LinearRegressionModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")