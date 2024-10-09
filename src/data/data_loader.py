from src.config.logging_config import setup_logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

logger = setup_logging()

class IngestData:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

def ingest_data(file_path: str) -> pd.DataFrame:
    try:
        ingest_data = IngestData(file_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise e

def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    from .data_pipeline import DataCleaning, DataPreprocessStrategy, DataDivideStrategy

    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise e