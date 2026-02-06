# api/main.py

from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List
import joblib

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0")

# Load trained Random Forest model and threshold
model = joblib.load("models/final_random_forest.pkl")
threshold = joblib.load("models/optimal_threshold.pkl")

# Pydantic model for input validation
class Transaction(BaseModel):
    features: List[float]  # Must contain exactly 30 numbers

# Example transaction for Swagger UI auto-fill
example_transaction = {
    "features": [
        0.0, -1.359807, -0.072781, 2.536346, 1.378155, -0.338321, 0.462388,
        0.239599, 0.098698, 0.363787, 0.090794, -0.5516, -0.617801, -0.99139,
        -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 0.403993,
        0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
        -0.189115, 0.133558, -0.021053, 149.62
    ]
}

@app.get("/")
def root():
    return {"message": "Fraud Detection API is up. Use /predict endpoint to get predictions."}

@app.post("/predict")
def predict(
    transaction: Transaction = Body(
        ...,
        example=example_transaction  # Swagger auto-fill
    )
):
    # Convert input list to 2D array for sklearn
    X = [transaction.features]

    # Predict fraud probability
    prob = model.predict_proba(X)[0][1]

    # Apply threshold
    is_fraud = int(prob >= threshold)

    # Return result
    return {
        "fraud_probability": round(prob, 3),
        "threshold": threshold,
        "is_fraud": is_fraud
    }
