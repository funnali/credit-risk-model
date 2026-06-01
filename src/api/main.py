import os
import sys
import numpy as np
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.api.pydantic_models import PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Bati Bank credit risk scoring model for buy-now-pay-later service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None


def load_model():
    """Load the best model from MLflow registry or local mlruns."""
    global model
    try:
        import mlflow
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("mlruns")
        model = mlflow.xgboost.load_model("models:/CreditRiskModel/1")
        logger.info("Model loaded from MLflow registry.")
    except Exception as e:
        logger.warning(f"Could not load from registry: {e}")
        logger.info("Loading fallback model from mlruns...")
        try:
            import mlflow.pyfunc
            runs = os.listdir("mlruns/0") if os.path.exists("mlruns/0") else []
            # Try loading latest xgboost model from mlruns
            for root, dirs, files in os.walk("mlruns"):
                for d in dirs:
                    model_path = os.path.join(root, d, "artifacts", "xgboost_model")
                    if os.path.exists(model_path):
                        model = mlflow.xgboost.load_model(model_path)
                        logger.info(f"Loaded model from {model_path}")
                        return
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            model = None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
def root():
    return {
        "message": "Credit Risk Prediction API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict credit risk probability for a customer.
    Returns risk_probability (0-1) and is_high_risk (0 or 1).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Convert booleans to int
        bool_cols = data.select_dtypes(include='bool').columns
        data[bool_cols] = data[bool_cols].astype(int)

        # Predict
        prob = float(model.predict_proba(data)[0][1])
        label = int(prob >= 0.5)
        risk_label = "HIGH RISK" if label == 1 else "LOW RISK"

        logger.info(f"Prediction: prob={prob:.4f}, label={risk_label}")

        return PredictionResponse(
            risk_probability=round(prob, 4),
            is_high_risk=label,
            risk_label=risk_label,
            model_version="1"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))