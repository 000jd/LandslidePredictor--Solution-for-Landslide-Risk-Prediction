from src.config.logging_config import setup_logging
import optuna

logger = setup_logging()

class HyperparameterTuner:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        logger.info(f"Initializing HyperparameterTuner for {model.__class__.__name__}")
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def objective(self, trial):
        try:
            return self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test)
        except Exception as e:
            logger.error(f"Error in optimization trial: {e}")
            return float('-inf')

    def optimize(self, n_trials=100):
        logger.info(f"Starting hyperparameter optimization for {n_trials} trials.")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        logger.info("Optimization completed.")
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        return best_params