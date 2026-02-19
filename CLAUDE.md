# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI-based ML inference service for a fine-tuned Iris classification model, deployed to Kubernetes via Devtron.

## Architecture

- `app/main.py` — FastAPI application with `/predict` (POST) and `/health` endpoints. Loads the pickled model at startup from `models/iris_finetuned_model.pkl` using joblib.
- `models/` — Contains the serialized scikit-learn model (`iris_finetuned_model.pkl`).
- `Dockerfile` — Multi-stage build using `python:3.11-slim`, runs as non-root user, exposes port 8000.
- `requirements.txt` — Pinned dependencies: fastapi, uvicorn, joblib, scikit-learn==1.17.0, numpy<2.0.0.

## Commands

### Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Build and run Docker
```bash
docker build -t flower-inference .
docker run -p 8000:8000 flower-inference
```

### Test endpoints
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Devtron Deployment

1. Push the Docker image to a container registry accessible by your Devtron cluster.
2. In Devtron, create a new application pointing to this repository.
3. Configure the build pipeline to use the `Dockerfile` at root.
4. Set the container port to **8000** in the deployment template.
5. Configure health checks using the `/health` endpoint.

## Key Constraints

- The model file (`iris_finetuned_model.pkl`) must be present in `models/` before building the Docker image.
- scikit-learn version must match the version used during model training (1.17.0) to avoid deserialization errors.
- numpy must be <2.0.0 for compatibility with the pinned scikit-learn version.
- The container runs as a non-root user — do not write to filesystem paths outside `/app` at runtime.
