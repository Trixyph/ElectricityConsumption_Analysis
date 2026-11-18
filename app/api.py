from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src', 'data'))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_v1.joblib')
THRESHOLD = 0.139

app = FastAPI(title="Smart Meter Anomaly API")

model = joblib.load(MODEL_PATH)

class MeterReading(BaseModel):
    timestamp: str = Field(..., example="2024-04-01 12:00:00")
    electricity_consumed: float = Field(..., ge=0, le=1)
    temperature: float = Field(..., ge=0, le=1)
    humidity: float = Field(..., ge=0, le=1)
    wind_speed: float = Field(..., ge=0, le=1)
    avg_past_consumption: float = Field(..., ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1", "threshold": THRESHOLD}

@app.post("/predict")
def predict(reading: MeterReading):
    try:
        df = pd.DataFrame([reading.model_dump()])
        df['Timestamp'] = pd.to_datetime(df['timestamp'])
        df.rename(columns={
            'electricity_consumed': 'Electricity_Consumed',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'wind_speed': 'Wind_Speed',
            'avg_past_consumption': 'Avg_Past_Consumption'
        }, inplace=True)
        
        proba = model.predict_proba(df)[0, 1]
        is_anomaly = bool(proba >= THRESHOLD)
        
        return {
            "timestamp": reading.timestamp,
            "probability": round(float(proba), 3),
            "is_anomaly": is_anomaly,
            "model_version": "v1"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))