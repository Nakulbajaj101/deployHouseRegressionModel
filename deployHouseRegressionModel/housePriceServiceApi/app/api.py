import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from houseregression_model import __version__ as model_version
from houseregression_model.predict import make_predictions
from loguru import logger

from app import schemas

api_router = APIRouter()
__version__ = "0.0.1"


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Root Get"""

    health = schemas.Health(
        name="House Model Prediction",
        api_version=__version__,
        model_version=model_version,
    )
    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults,
                 status_code=200)
async def predict(input_data: schemas.MultipleHouseDataInputs) -> Any:
    """Make a house prediction"""

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making predictions for inputs: {input_data.inputs}")
    results = make_predictions(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        detail = json.dumps(results.get("errors"))
        raise HTTPException(status_code=400, detail=json.loads(detail))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
