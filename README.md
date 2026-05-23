# Transaction Category Classification AI

AI-powered bank transaction narration classification system using:

- FastAPI
- DistilBERT
- Transformers
- Batch Inference

## Features

- Fast transaction category prediction
- Batch processing support
- Optimized transformer inference
- Render deployment ready
- Spring Boot integration ready

## Tech Stack

- Python
- FastAPI
- Hugging Face Transformers
- DistilBERT
- Render

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Deployment

Deployed using Render.

## API Endpoint

```text
POST /predict-category
```

## Example Request

```json
{
  "narrations": [
    "SWIGGY PAYMENT",
    "AMAZON SHOPPING"
  ]
}
```