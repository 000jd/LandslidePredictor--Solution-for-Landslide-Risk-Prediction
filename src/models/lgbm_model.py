import logging
from lightgbm import LGBMRegressor
from src.models.base_model import Model

class LightGBMModel(Model):
    def train(self, x_train, y_train, **kwargs):
        logging.info("Training LightGBM model...")
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logging.info("Optimizing LightGBM model...")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(x_test, y_test)