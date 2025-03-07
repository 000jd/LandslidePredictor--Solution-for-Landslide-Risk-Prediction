import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import logging
from fastapi import Depends, Request

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.current_model_type = None
        self.expected_features = [
            'elevation', 'slope', 'aspect', 'rainfall_daily', 'rainfall_monthly',
            'distance_to_faults', 'soil_depth', 'vegetation_density', 
            'earthquake_magnitude', 'soil_moisture', 'previous_landslides',
            'snow_melt', 'landslide_probability',
            'lithology_basalt', 'lithology_granite', 'lithology_limestone', 
            'lithology_sandstone', 'lithology_shale',
            'land_use_agriculture', 'land_use_barren', 'land_use_forest', 
            'land_use_grassland', 'land_use_urban',
            'human_activity_high', 'human_activity_low', 'human_activity_medium'
        ]

    def load_model(self, model_path: str = "saved_models/xgboost_20250306_193033.pkl"):
        try:
            self.model = joblib.load(Path(model_path))
            self.current_model_type = Path(model_path).stem.split('_')[0]
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
            else:
                self.feature_names = self.expected_features
                
            logger.info(f"Loaded {self.current_model_type} model with features: {self.feature_names}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables
        categorical_mappings = {
            'lithology': ['basalt', 'granite', 'limestone', 'sandstone', 'shale'],
            'land_use': ['agriculture', 'barren', 'forest', 'grassland', 'urban'],
            'human_activity': ['high', 'low', 'medium']
        }
        
        for field, categories in categorical_mappings.items():
            value = input_data[field]
            for category in categories:
                df[f"{field}_{category}"] = 1 if value == category else 0
                
        # Ensure all expected features exist
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0
                
        return df[self.feature_names]

    def predict(self, input_data: Dict) -> float:
        if not self.model:
            raise ValueError("No model loaded")
            
        try:
            processed_data = self._preprocess_input(input_data)
            prediction = self.model.predict(processed_data)
            return float(np.clip(prediction[0], 0, 1))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service