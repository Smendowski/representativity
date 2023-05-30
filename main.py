from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import services
from data.models import Dataset, Sample
from exceptions import (
    InferenceSampleHasUnexpectedShapeError,
    ModelNotFittedError
)

app = FastAPI()


@app.post("/train")
async def train_model(dataset: Dataset, background_tasks: BackgroundTasks) -> JSONResponse:
    try:
        background_tasks.add_task(services.train_model, dataset)
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "detail": "Job has been submitted"
        }
    )


@app.post("/predict")
async def get_model_prediction(samples: list[Sample]):
    representativeness = []
    try:
        for sample in samples:
            representativeness.append(await services.get_model_prediction(sample))
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )
    except (ModelNotFittedError, InferenceSampleHasUnexpectedShapeError) as error:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=str(error)
        )
    return jsonable_encoder({
            "representativeness": representativeness
    })


@app.get("/status")
async def get_model_status():
    model_status: dict[str, str] = await services.get_model_status()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=model_status
    )
