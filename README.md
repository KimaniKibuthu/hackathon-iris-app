# hackathon-iris-app

This repository contains a FastAPI application for classifying iris flowers using a pre-trained machine learning model.

## Contents

1. FastAPI application for iris classification (`iris_app.py`)
2. Pre-trained SVC model (`svc_model.pkl`)

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn (for running the FastAPI application)
- Scikit-learn (for the SVC model)
- Other dependencies (list them here)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/iris-classification.git
   cd iris-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

2. Open your browser and go to `http://localhost:8000/docs` to access the Swagger UI and test the API.
