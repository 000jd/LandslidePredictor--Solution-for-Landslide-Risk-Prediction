import logging
from sklearn.linear_model import LinearRegression
from src.models.base_model import Model

class LinearRegressionModel(Model):
    def train(self, x_train, y_train, **kwargs):
        logging.info("Training LinearRegression model...")
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        logging.info("Optimizing LinearRegression model...")
        # No hyperparameter tuning necessary for Linear Regression
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)