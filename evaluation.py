import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ===========================
# 1️⃣ Evaluate Transaction Model
# ===========================

def evaluate_transaction():
    print("\n--- Evaluating Transaction Model ---")

    df = pd.read_csv("1_transactions_DL/data/online_payments.csv")

    if "isFraud" not in df.columns:
        print("Transaction dataset does not contain isFraud column.")
        return

    X = torch.tensor(df.drop("isFraud", axis=1).values, dtype=torch.float32)
    y_true = df["isFraud"].astype(int).values

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )

    model_path = "1_transactions_DL/transaction_model.pt"
    if not os.path.exists(model_path):
        print("Transaction model not found.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(X).squeeze().numpy()

    y_pred = (preds > 0.5).astype(int)

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))


# ===========================
# 2️⃣ Evaluate Complaint NLP Model
# ===========================

def evaluate_complaint():
    print("\n--- Evaluating Complaint NLP Model ---")

    dataset_path = "2_complaints_NLP/data/Complaints.csv"
    model_path = "2_complaints_NLP/complaint_model"

    if not os.path.exists(dataset_path):
        print("Complaint dataset not found.")
        return

    df = pd.read_csv(dataset_path)

    if "Consumer complaint narrative" not in df.columns:
        print("Required complaint column missing.")
        return

    df = df[["Consumer complaint narrative"]].dropna().head(200)

    texts = df["Consumer complaint narrative"].tolist()
    labels = [1]*len(texts)  # dummy labels (for demo)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=1).numpy()

    print("Accuracy :", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall   :", recall_score(labels, preds))
    print("F1 Score :", f1_score(labels, preds))


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    evaluate_transaction()
    evaluate_complaint()
