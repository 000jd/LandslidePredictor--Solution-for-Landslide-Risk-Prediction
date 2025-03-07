from src.config.logging_config import setup_logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logger = setup_logging()

EXPECTED_FEATURES = [
    'elevation', 'slope', 'aspect', 'rainfall_daily', 'rainfall_monthly',
    'distance_to_faults', 'soil_depth', 'vegetation_density', 'earthquake_magnitude',
    'soil_moisture', 'previous_landslides', 'snow_melt', 'landslide_probability',
    'lithology_basalt', 'lithology_granite', 'lithology_limestone', 
    'lithology_sandstone', 'lithology_shale',
    'land_use_agriculture', 'land_use_barren', 'land_use_forest', 
    'land_use_grassland', 'land_use_urban',
    'human_activity_high', 'human_activity_low', 'human_activity_medium'
]

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Available columns: {data.columns.tolist()}")
            
            # Drop unnecessary columns if they exist
            columns_to_drop = ["date", "latitude", "longitude"]
            data = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)
            
            # Identify categorical and numeric columns
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            
            logger.info(f"Categorical columns: {categorical_columns.tolist()}")
            logger.info(f"Numeric columns: {numeric_columns.tolist()}")
            
            # Identify target variable (assuming it's the last column)
            target_column = 'landslide_occurred'
            logger.info(f"Target variable is: {target_column}")
            
            # Separate features and target
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_columns.drop(target_column)),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                ])
            
            # Fit and transform the data
            X_transformed = preprocessor.fit_transform(X)
            
            # Get feature names after preprocessing
            numeric_feature_names = numeric_columns.drop(target_column).tolist()
            categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns).tolist()
            feature_names = numeric_feature_names + categorical_feature_names
            
            # Convert to DataFrame
            X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
            
            logger.info(f"Preprocessed data shape: {X_transformed_df.shape}")
            logger.info(f"Target variable shape: {y.shape}")
            
            return X_transformed_df, y
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: Tuple[pd.DataFrame, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            X, y = data
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
