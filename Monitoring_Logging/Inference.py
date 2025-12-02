# Monitoring dan Logging/7.inference.py
import os
import json
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# CONFIG: ubah sesuai lokasi modelmu
MODEL_URI = os.environ.get("MODEL_URI", "mlruns/0/61a2cde5c7f74ca99b741f8c9c9ff59e/artifacts/model")

# METRICS
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "Request latency", ["endpoint"])

# Load model (pyfunc) once
print("Loading model from:", MODEL_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded.")

app = FastAPI(title="Model Inference")

class PredictRequest(BaseModel):
    data: list  # list of records (list of lists) or list of dicts

@app.post("/predict")
async def predict(req: PredictRequest):
    endpoint = "/predict"
    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        try:
            # Accepts list of lists -> convert to DataFrame; if list of dicts, pd.DataFrame handles it
            if len(req.data) == 0:
                raise ValueError("data is empty")
            if isinstance(req.data[0], list):
                df = pd.DataFrame(req.data)
            else:
                df = pd.DataFrame(req.data)
            preds = model.predict(df)
            REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
            # convert to plain python types
            return {"predictions": preds.tolist()}
        except Exception as e:
            REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
            return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    # Expose prometheus metrics
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Jalankan dev server
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
