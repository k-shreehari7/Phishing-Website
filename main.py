
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import predict_url

from predict import predict_url

HTML_FILE='index.html'
app = FastAPI()


@app.get("/")
def home():
    return FileResponse(HTML_FILE)


# =========================================================
# REQUEST FORMAT
# =========================================================

class PredictionRequest(BaseModel):

    url: str
    model: str


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        result = predict_url(
            request.url,
            request.model
        )

        return {
            "url": request.url,
            "model": result["model"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "phishing_probability":
                result["phishing_probability"]
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__== "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)