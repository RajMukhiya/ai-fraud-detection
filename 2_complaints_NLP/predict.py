import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use absolute path for model loading
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Note: Nested structure observed in complaint_model directory
MODEL_PATH = os.path.join(BASE_PATH, "complaint_model/complaint_model")

def predict_complaint(text):
    # Ensure model path is valid
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    model.eval()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    fraud_score = probs[0][1].item()
    
    # Check for NaN and return a reasonable default if necessary
    import math
    if math.isnan(fraud_score):
        return 0.5

    return fraud_score


if __name__ == "__main__":
    test_text = "Money was deducted without my permission"
    score = predict_complaint(test_text)
    print("Complaint Fraud Score:", round(score, 3))
