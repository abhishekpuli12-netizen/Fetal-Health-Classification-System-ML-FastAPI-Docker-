import joblib
import os

def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "artifacts", "model.pkl")

    print("Loading model from:", model_path)

    return joblib.load(model_path)