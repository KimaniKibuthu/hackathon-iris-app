from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import uvicorn
import logging

# Load the model (better with try-except for error handling)
try:
    model = joblib.load('svc_model.pkl')
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise RuntimeError("Could not load the model. Ensure 'svc_model.pkl' is available.")

# Create the FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="A simple API that classifies iris flowers based on sepal and petal measurements.",
    version="1.0.0"
)

# Define the request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint to predict iris species
@app.post("/predict", summary="Predict Iris Species", tags=["Prediction"])
async def predict(features: IrisFeatures):
    try:
        # Log incoming request
        logging.info(f"Received request with features: {features}")
        
        # Prepare the data for prediction
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Perform the prediction
        prediction = model.predict(input_data)
        prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        # Return the response with meaningful prediction
        return {
            "prediction": prediction_map[int(prediction[0])]
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Health check endpoint
@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check():
    return {"status": "Healthy"}

# Run the app
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
