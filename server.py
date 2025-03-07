'''
import numpy as np
import pandas as pd
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default feature list (fallback if model lacks feature_names_in_)
EXPECTED_FEATURES = [
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

# Categories for one-hot encoding
lithology_categories = ['basalt', 'granite', 'limestone', 'sandstone', 'shale']
land_use_categories = ['agriculture', 'barren', 'forest', 'grassland', 'urban']
human_activity_categories = ['high', 'low', 'medium']

# Initialize FastAPI app
app = FastAPI()

# Global variables
model = None
expected_features = None

# Input data schema
class InputData(BaseModel):
    elevation: float
    slope: float
    aspect: float
    rainfall_daily: float
    rainfall_monthly: float
    distance_to_faults: float
    soil_depth: float
    vegetation_density: float
    earthquake_magnitude: float
    soil_moisture: float
    previous_landslides: int
    snow_melt: float
    landslide_probability: float
    lithology: str
    land_use: str
    human_activity: str

# Load model and feature names at startup
@app.on_event("startup")
def load_model_func():
    global model, expected_features
    model_path = Path("saved_models/xgboost_20250306_193033.pkl")  # Adjust path as needed
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            logger.info(f"Model expects features: {expected_features}")
        else:
            expected_features = EXPECTED_FEATURES
            logger.warning("Model lacks feature_names_in_, using default EXPECTED_FEATURES")
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Model loading failed")

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Numerical features
        numerical_features = {
            'elevation': data.elevation,
            'slope': data.slope,
            'aspect': data.aspect,
            'rainfall_daily': data.rainfall_daily,
            'rainfall_monthly': data.rainfall_monthly,
            'distance_to_faults': data.distance_to_faults,
            'soil_depth': data.soil_depth,
            'vegetation_density': data.vegetation_density,
            'earthquake_magnitude': data.earthquake_magnitude,
            'soil_moisture': data.soil_moisture,
            'previous_landslides': data.previous_landslides,
            'snow_melt': data.snow_melt,
            'landslide_probability': data.landslide_probability
        }

        # Validate categorical inputs
        if data.lithology not in lithology_categories:
            raise ValueError(f"Invalid lithology: {data.lithology}")
        if data.land_use not in land_use_categories:
            raise ValueError(f"Invalid land_use: {data.land_use}")
        if data.human_activity not in human_activity_categories:
            raise ValueError(f"Invalid human_activity: {data.human_activity}")

        # One-hot encode categorical features
        lithology_encoded = {f'lithology_{cat}': 1 if cat == data.lithology else 0 
                             for cat in lithology_categories}
        land_use_encoded = {f'land_use_{cat}': 1 if cat == data.land_use else 0 
                            for cat in land_use_categories}
        human_activity_encoded = {f'human_activity_{cat}': 1 if cat == data.human_activity else 0 
                                  for cat in human_activity_categories}

        # Combine all features
        features = {**numerical_features, **lithology_encoded, **land_use_encoded, **human_activity_encoded}

        # Create DataFrame and reorder to match model's expected features
        df = pd.DataFrame([features])[expected_features]

        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(df)[0][1]  # Probability of positive class
        else:
            prediction = model.predict(df)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()

        return {"prediction": float(prediction)}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
''' 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routers.predictions import router as predictions_router
from api.services.model_service import ModelService
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model service instance
    model_service = ModelService()
    try:
        model_service.load_model()
        app.state.model_service = model_service  
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError("Model initialization failed")
    yield
    # Clean up on shutdown
    app.state.model_service = None

app = FastAPI(
    title="Landslide Prediction API",
    description="API for predicting landslide risks using machine learning models",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions_router, prefix="/api/v1", tags=["predictions"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model-health")
async def model_health():
    return {
        "model_loaded": app.state.model_service is not None,
        "model_type": app.state.model_service.current_model_type if app.state.model_service else None
    }