from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    model_name: str = Field(..., example="xgboost_20250306_193033")
    elevation: float = Field(..., ge=0, le=10000)
    slope: float = Field(..., ge=0, le=90)
    aspect: float = Field(..., ge=0, le=360)
    rainfall_daily: float = Field(..., ge=0)
    rainfall_monthly: float = Field(..., ge=0)
    distance_to_faults: float = Field(..., ge=0)
    soil_depth: float = Field(..., ge=0)
    vegetation_density: float = Field(..., ge=0, le=1)
    earthquake_magnitude: float = Field(..., ge=0)
    soil_moisture: float = Field(..., ge=0, le=100)
    previous_landslides: int = Field(..., ge=0)
    snow_melt: float = Field(..., ge=0)
    landslide_probability: float = Field(..., ge=0, le=1)
    lithology: str
    land_use: str
    human_activity: str

class PredictionOutput(BaseModel):
    prediction: float = Field(..., ge=0, le=1)