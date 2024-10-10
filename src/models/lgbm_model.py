from lightgbm import LGBMRegressor
import numpy as np
from src.config.logging_config import setup_logging
from src.models.base_model import Model
from typing import Dict, Any

logger = setup_logging()

class LightGBMModel(Model):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def train(self, x_train, y_train, **kwargs):
        logger.info("Training LightGBM model...")
        train_config = {**self.config, **kwargs}
        reg = LGBMRegressor(**train_config)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logger.info("Optimizing LightGBM model...")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_float("learning_rate", 1e-7, 1.0, log=True)
        
        try:
            reg = self.train(x_train, y_train, 
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            learning_rate=learning_rate)
            
            score = reg.score(x_test, y_test)
            
            if not np.isfinite(score):
                return float('-inf')
            return score
        except Exception as e:
            logger.error(f"Error during LightGBM optimization: {e}")
            return float('-inf')