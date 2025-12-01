from fastapi import FastAPI
import mlflow
import numpy as np

# Load model dari MLflow
model = mlflow.keras.load_model("mlruns/0/d6fa3809d53043c3a7bae994aba767a0/artifacts/model")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Model Serving is running..."}

@app.get("/predict")
def predict(value: float):
    data = np.array([[value]])
    prediction = model.predict(data)
    return {"prediction": float(prediction[0][0])}
