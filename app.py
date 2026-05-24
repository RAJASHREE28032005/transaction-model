from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

import torch
import time

print("STARTING FASTAPI APP")

app = FastAPI()

MODEL_PATH = "RAJASHREE28032005/transaction-category-model"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded!")

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32
)

print("Model loaded successfully!")

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

print("Pipeline created successfully!")

class Transaction(BaseModel):
    transactionId: int
    narration: str

class PredictionRequest(BaseModel):
    transactions: list[Transaction]

@app.get("/")
def home():

    return {
        "status": "running"
    }

@app.post("/predict-category")
def predict(request: PredictionRequest):

    start_time = time.time()

    narrations = [
        txn.narration
        for txn in request.transactions
    ]

    predictions = classifier(
        narrations,
        batch_size=32
    )

    response = []

    for txn, pred in zip(
        request.transactions,
        predictions
    ):

        response.append({
            "transactionId": txn.transactionId,
            "category": pred["label"],
            "confidence": round(
                pred["score"],
                2
            )
        })

    end_time = time.time()

    return {
        "predictions": response,
        "processing_time_seconds":
            round(end_time - start_time, 2)
    }