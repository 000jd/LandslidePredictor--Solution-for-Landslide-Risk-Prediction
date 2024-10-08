import logging

def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {str(e)}")
            raise
    return wrapper