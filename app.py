from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load(
    "models/nifty_best_model.pkl"
)

class NiftyInput(BaseModel):
    Return_1d: float
    Return_3d: float
    Return_5d: float
    MA_5: float
    MA_10: float
    MA_20: float
    Volatility_5: float
    Volatility_10: float

@app.get("/")
def home():
    return {"message": "Nifty Predictor API Running"}

@app.post("/predict")
def predict(data: NiftyInput):

    features = pd.DataFrame([{
        "Return_1d": data.Return_1d,
        "Return_3d": data.Return_3d,
        "Return_5d": data.Return_5d,
        "MA_5": data.MA_5,
        "MA_10": data.MA_10,
        "MA_20": data.MA_20,
        "Volatility_5": data.Volatility_5,
        "Volatility_10": data.Volatility_10
    }])

    prediction = model.predict(features)[0]

    return {
        "prediction": "UP" if prediction == 1 else "DOWN"
    }