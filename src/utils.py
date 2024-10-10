from datetime import datetime
import os
import numpy as np
from src.config.logging_config import setup_logging
logger = setup_logging()

def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            raise
    return wrapper

def save_model(model, model_name):
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    import joblib
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def check_data(X, y, dataset_name):
    logger.info(f"Checking {dataset_name} dataset...")
    logger.info(f"{dataset_name} X shape: {X.shape}")
    logger.info(f"{dataset_name} y shape: {y.shape}")
    
    # Check for NaN values
    if X.isnull().any().any():
        logger.warning(f"NaN values found in {dataset_name} features")
    if y.isnull().any():
        logger.warning(f"NaN values found in {dataset_name} target")
    
    # Check numeric columns for non-finite values
    if not np.isfinite(X).all().all():
        logger.warning(f"Non-finite values found in {dataset_name} features")
    
    # Check target variable
    if not np.isfinite(y).all():
        logger.warning(f"Non-finite values found in {dataset_name} target")