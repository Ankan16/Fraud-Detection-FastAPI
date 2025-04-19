# Fraud Detection API

A machine learning-powered API for real-time fraud detection of financial transactions.

## Overview

This project provides a REST API built with FastAPI that uses a pre-trained machine learning model to detect potentially fraudulent transactions. It includes both API endpoints for integration and a simple UI for demonstration purposes.

## Features

- **Machine Learning Model**: Uses a pre-trained model to predict fraudulent transactions
- **REST API**: Fast and reliable endpoints built with FastAPI
- **Interactive UI**: Simple web interface to test the model in real-time
- **Docker Support**: Easily deployable using Docker containers

## Quick Start

### Using Docker

```bash
# Build the Docker image
docker build -t fraud-detection .

# Run the container
docker run -p 8000:8000 fraud-detection
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --reload
```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Submit transaction features for fraud prediction
- `GET /ui`: Web interface for testing the model

## Example Usage

### API Request

```python
import requests

# Test data
transaction = {
    "features": [0.2, 0.5, -0.1, 0.8]  # Replace with actual feature values
}

# Send request
response = requests.post("http://localhost:8000/predict", json=transaction)
result = response.json()

print(f"Prediction: {'Fraudulent' if result['prediction'] == 1 else 'Genuine'}")
print(f"Probability: {result['probability']:.2f}")
```

### Web Interface

Navigate to `http://localhost:8000/ui` in your browser to access the interactive testing interface.

## Project Structure

```
fraud-detection-api/
├── main.py              # FastAPI application
├── requirements.txt     # Project dependencies
├── Dockerfile           # Docker configuration
├── fraud_model.pkl      # Pre-trained ML model
└── templates/           # UI templates
    └── index.html       # Web interface
```

## Requirements

- Python 3.11+
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- NumPy
- Joblib
- jinja2

## License

[MIT License](LICENSE)
