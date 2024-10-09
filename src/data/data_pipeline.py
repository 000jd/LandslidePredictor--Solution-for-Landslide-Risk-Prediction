from src.config.logging_config import setup_logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = setup_logging()

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Drop unnecessary columns
            data = data.drop(["date", "latitude", "longitude"], axis=1)
            
            # Convert categorical features to 'category' dtype
            categorical_columns = ['land_use', 'lithology', 'human_activity']
            for col in categorical_columns:
                data[col] = data[col].astype('category')
            
            # Handle NaN values
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
            
            # For categorical columns, fill NaN with the most frequent value
            for col in categorical_columns:
                data[col] = data[col].fillna(data[col].mode()[0])
            
            return data
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            features = [
                'elevation', 'slope', 'aspect', 'rainfall_daily', 'rainfall_monthly',
                'distance_to_faults', 'soil_depth', 'vegetation_density', 'land_use',
                'lithology', 'earthquake_magnitude', 'soil_moisture', 'human_activity',
                'previous_landslides', 'snow_melt', 'landslide_probability',
            ]
            target = 'landslide_occurred'

            X = data[features]
            y = data[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in dividing data: {e}")
            raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.df)

class FeatureConfig:
    def __init__(self):
        self.numeric_features = [
            'elevation', 'slope', 'aspect', 'rainfall_daily', 'rainfall_monthly',
            'distance_to_faults', 'soil_depth', 'vegetation_density', 'earthquake_magnitude',
            'soil_moisture', 'previous_landslides', 'snow_melt', 'landslide_probability',
        ]
        self.categorical_features = ['lithology', 'land_use', 'human_activity']

def create_preprocessor():
    features = FeatureConfig()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features.numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), features.categorical_features)
        ]
    )

    return preprocessor