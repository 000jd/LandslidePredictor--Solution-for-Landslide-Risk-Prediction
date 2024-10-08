class ModelNameConfig:
    def __init__(self, model_name: str, fine_tuning: bool = True):
        self.model_name = model_name
        self.fine_tuning = fine_tuning

MODEL_CONFIGS = {
    "randomforest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
    },
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 10,
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 10,
    },
    "linear_regression": {},
}