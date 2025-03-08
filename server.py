# server.py (main FastAPI file)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from api.routers.predictions import router as predictions_router
from api.services.model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model service with correct paths
    app.state.model_service = ModelService()
    app.state.model_service.model_dir = Path("saved_models")  # Point to outer saved_models
    try:
        # Load default model or handle empty state
        models = app.state.model_service.get_available_models()
        if models:
            logger.info(f"Available models: {models}")
        else:
            logger.warning("No models found in saved_models directory")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
    yield
    # Cleanup on shutdown
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

# Include routers from API folder
app.include_router(predictions_router, prefix="/api/v1", tags=["predictions"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model-health")
async def model_health(request: Request):
    service = request.app.state.model_service
    return {
        "loaded_models": list(service.models.keys()) if service else [],
        "available_models": service.get_available_models() if service else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)