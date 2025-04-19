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

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

