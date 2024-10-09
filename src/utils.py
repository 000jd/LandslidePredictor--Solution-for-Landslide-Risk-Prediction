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