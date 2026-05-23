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
# STARTUP LOGS
# =========================================

print("STARTING FASTAPI APP")

# =========================================
# FASTAPI APP
# =========================================

app = FastAPI()

# =========================================
# MODEL PATH
# =========================================

MODEL_PATH = "./saved_model"

# =========================================
# LOAD TOKENIZER
# =========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded!")

# =========================================
# LOAD MODEL
# =========================================

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

print("Model loaded successfully!")

# =========================================
# FORCE CPU
# =========================================

device = -1

print("Using CPU device")

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

print("Pipeline created successfully!")

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
# EXTRA HEALTH ROUTE
# =========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# =========================================
# PREDICT CATEGORY
# =========================================

@app.post("/predict-category")
def predict(request: PredictionRequest):

    start_time = time.time()

    print(
        f"Received {len(request.transactions)} transactions"
    )

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
        f"Processed {len(response)} "
        f"transactions in {total_time} sec"
    )

    return {
        "predictions": response,
        "processing_time_seconds": total_time
    }

# =========================================
# STARTUP COMPLETE
# =========================================

print("FASTAPI APP STARTUP COMPLETE")