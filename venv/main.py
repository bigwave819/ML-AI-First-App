from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

model = None

# 📥 Try to load the model safely
if os.path.exists("model.joblib"):
    model = joblib.load("model.joblib")
    print("✅ Model loaded successfully.")
else:
    print("⚠️ model.joblib file not found! Please train the model first using train_model.py")

# 📦 Define request body structure
class StudyHours(BaseModel):
    hours: float

# 🎯 Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Student Score Predictor API"}

# 🔍 Prediction route
@app.post("/predict")
def predict_score(data: StudyHours):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")

    hours = np.array([[data.hours]])
    prediction = model.predict(hours)
    return {
        "hours": data.hours,
        "predicted_score": round(prediction[0], 2)
    }
