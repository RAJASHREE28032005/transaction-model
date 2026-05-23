from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

import torch
import time

# =========================================
# FASTAPI APP
# =========================================

app = FastAPI()

# =========================================
# LOAD MODEL ONLY ONCE
# =========================================

MODEL_PATH = "./saved_model"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# =========================================
# DEVICE CONFIG
# =========================================

device = 0 if torch.cuda.is_available() else -1

print("Using device:", device)

# =========================================
# CREATE PIPELINE
# =========================================

classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    truncation=True,
    max_length=64
)

print("Model loaded successfully!")

# =========================================
# REQUEST MODELS
# =========================================

class Transaction(BaseModel):
    transactionId: int
    narration: str

class PredictionRequest(BaseModel):
    transactions: list[Transaction]

# =========================================
# HEALTH CHECK
# =========================================

@app.get("/")
def home():
    return {
        "status": "AI service running"
    }

# =========================================
# PREDICT CATEGORY
# =========================================

@app.post("/predict-category")
def predict(request: PredictionRequest):

    start_time = time.time()

    # =====================================
    # EXTRACT NARRATIONS
    # =====================================

    narrations = [
        txn.narration
        for txn in request.transactions
    ]

    # =====================================
    # BATCH INFERENCE
    # =====================================

    predictions = classifier(
        narrations,
        batch_size=32
    )

    # =====================================
    # BUILD RESPONSE
    # =====================================

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

    total_time = round(
        end_time - start_time,
        2
    )

    print(
        f"Processed {len(request.transactions)} "
        f"transactions in {total_time} sec"
    )

    return {
        "predictions": response
    }