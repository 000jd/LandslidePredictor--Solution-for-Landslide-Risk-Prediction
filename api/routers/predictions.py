from fastapi import APIRouter, Depends, HTTPException, Request
from api.services.model_service import ModelService, get_model_service
from api.models.schemas import PredictionInput, PredictionOutput
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionOutput)
async def predict(
    data: PredictionInput,
    request: Request,
    model_service: ModelService = Depends(get_model_service)
):
    try:
        prediction = model_service.predict(data.dict())
        return {"prediction": prediction}
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")