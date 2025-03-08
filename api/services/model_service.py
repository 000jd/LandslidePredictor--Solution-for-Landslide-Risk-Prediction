import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import logging
from fastapi import Depends, Request

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.models = {}  # Dictionary to cache loaded models
        self.current_model = None
        self.model_dir = Path(__file__).parent.parent / "saved_models"
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
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return [f.stem for f in self.model_dir.glob("*.pkl")]

    def load_model(self, model_name: str):
        """Load or get cached model"""
        if model_name in self.models:
            return self.models[model_name]
        
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
            
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            raise
        
    def _preprocess_input(self, input_data: Dict, model) -> pd.DataFrame:
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Get expected features from model
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
        else:
            expected_features = self.expected_features
        
        # One-hot encode categorical variables
        categorical_mappings = {
            'lithology': ['basalt', 'granite', 'limestone', 'sandstone', 'shale'],
            'land_use': ['agriculture', 'barren', 'forest', 'grassland', 'urban'],
            'human_activity': ['high', 'low', 'medium']
        }
        
        for field, categories in categorical_mappings.items():
            value = input_data.get(field)
            if not value:
                raise ValueError(f"Missing required field: {field}")
                
            for category in categories:
                df[f"{field}_{category}"] = 1 if value == category else 0
        
        # Add missing features with 0 values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Keep only expected features and order them
        return df[expected_features]

    def predict(self, model_name: str, input_data: Dict) -> float:
        """Make prediction with specified model"""
        try:
            model = self.load_model(model_name)
            processed_data = self._preprocess_input(input_data, model)
            logger.info(f"Processed data columns: {processed_data.columns.tolist()}")
            prediction = model.predict(processed_data)
            return float(np.clip(prediction[0], 0, 1))
        except Exception as e:
            logger.exception("Prediction failed with error:")  # Log full traceback
            raise

def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service