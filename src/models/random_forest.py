from src.config.logging_config import setup_logging
from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import Model
import numpy as np
from typing import Dict, Any

logger = setup_logging()

class RandomForestModel(Model):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def train(self, x_train, y_train, **kwargs):
        logger.info("Training RandomForest model...")
        train_config = {**self.config, **kwargs}
        reg = RandomForestRegressor(**train_config)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logger.info("Optimizing RandomForest model...")
        
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        
        try:
            reg = self.train(x_train, y_train, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        min_samples_split=min_samples_split, 
                        min_samples_leaf=min_samples_leaf)
            
            score = reg.score(x_test, y_test)
            
            if not np.isfinite(score):
                return float('-inf')
            return score
        except Exception as e:
            logger.error(f"Error during RandomForest optimization: {e}")
            return float('-inf')