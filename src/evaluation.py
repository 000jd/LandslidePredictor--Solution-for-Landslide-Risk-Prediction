from src.config.logging_config import setup_logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

logger = setup_logging()

class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class MSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            mse = mean_squared_error(y_true, y_pred)
            logger.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logger.error(f"Error calculating MSE: {e}")
            raise e

class R2Score(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            r2 = r2_score(y_true, y_pred)
            logger.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logger.error(f"Error calculating R2 Score: {e}")
            raise e

class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logger.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}")
            raise e