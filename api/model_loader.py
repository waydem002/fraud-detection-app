import joblib

MODEL_PATH = "models/rf_smote_tuned.pkl"
THRESHOLD_PATH = "models/optimal_threshold.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold
