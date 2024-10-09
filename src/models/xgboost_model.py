from src.config.logging_config import setup_logging
import xgboost as xgb
from src.models.base_model import Model
import numpy as np 

logger = setup_logging()
class XGBoostModel(Model):
    def train(self, x_train, y_train, **kwargs):
        logger.info("Training XGBoost model...")
        reg = xgb.XGBRegressor(**kwargs, enable_categorical=True)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logger.info("Optimizing XGBoost model...")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_float("learning_rate", 1e-7, 1.0, log=True)
        
        try:
            reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            score = reg.score(x_test, y_test)
            if not np.isfinite(score):
                return float('-inf')
            return score
        except Exception as e:
            logger.error(f"Error during XGBoost optimization: {e}")
            return float('-inf')