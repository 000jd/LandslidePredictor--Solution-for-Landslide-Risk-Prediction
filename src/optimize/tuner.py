import logging
import optuna

class HyperparameterTuner:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        logging.info(f"Initializing HyperparameterTuner for {model.__class__.__name__}")
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        logging.info(f"Starting hyperparameter optimization for {n_trials} trials.")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        logging.info("Optimization completed.")
        return study.best_trial.params