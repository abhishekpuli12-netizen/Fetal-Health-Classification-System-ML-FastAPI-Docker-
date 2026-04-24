from fastapi import FastAPI, HTTPException
import numpy as np
from src.predict import load_model
from app.schema import FetalData

# ✅ define app FIRST
app = FastAPI()

# load model
model = load_model()

# feature order
FEATURES = [
    "baseline_value", "accelerations", "fetal_movement",
    "uterine_contractions", "light_decelerations",
    "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max",
    "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median",
    "histogram_variance", "histogram_tendency"
]

@app.get("/")
def home():
    return {"message": "Fetal Health API Running"}

@app.post("/predict")
def predict(data: FetalData):
    try:
        input_data = np.array([[getattr(data, f) for f in FEATURES]])

        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]

        labels = ["Normal", "Suspect", "Pathological"]

        return {
            "prediction": int(prediction + 1),
            "label": labels[prediction],
            "probabilities": probs.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))