from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")

app = FastAPI()

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "It works!"}

@app.post("/predict")
def predict(data: Transaction):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    return {
        "prediction": int(prediction),
        "fraud_probability": round(probability, 4)
    }



