from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction.predict_disease import predict_disease


app = FastAPI(
    title="Medical AI Disease Prediction API",
    description=(
        "AI-based disease prediction system using "
        "medical text classification."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):

    symptoms: str = Field(
        ...,
        min_length=3,
        description="Patient symptoms",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of ranked predictions",
    )


@app.get("/")
def root():

    return {
        "project": "Medical AI Disease Prediction",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # predict_disease() expects the medical text
        # as its first positional argument.
        result = predict_disease(
            request.symptoms,
            request.top_k,
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {error}",
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )