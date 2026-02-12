from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from pydantic import BaseModel
import os
import sys
import importlib.util

app = FastAPI(title="AI Fraud Detection API")

# ===========================
# Dynamic Module Loader
# ===========================
def load_module_from_path(module_name, file_path):
    module_dir = os.path.dirname(file_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models dynamically to reuse logic from predict.py
try:
    predict_dl = load_module_from_path("predict_dl", os.path.join(BASE_DIR, "1_transactions_DL", "predict.py"))
    predict_nlp = load_module_from_path("predict_nlp", os.path.join(BASE_DIR, "2_complaints_NLP", "predict.py"))
    predict_cv = load_module_from_path("predict_cv", os.path.join(BASE_DIR, "3_kyc_CV", "predict.py"))
    fusion_engine = load_module_from_path("fusion_engine", os.path.join(BASE_DIR, "4_fusion_engine", "fraud_score.py"))
except Exception as e:
    print(f"Error loading model modules: {e}")
    # We'll handle missing modules in the endpoint if necessary

# ===========================
# Request Schema
# ===========================
class TransactionData(BaseModel):
    amount: float
    type: str
    oldbalanceOrg: float
    newbalanceOrig: float

class FraudRequest(BaseModel):
    transaction: TransactionData
    complaint_text: str
    selfie_path: str = "selfie.jpg"
    id_path: str = "id.jpg"

# ===========================
# Prediction Endpoint
# ===========================
@app.post("/predict")
def predict_fraud(request: FraudRequest):
    try:
        # 1. Transaction score
        t_score = predict_dl.predict_transaction(request.transaction.model_dump())

        # 2. Complaint score
        c_score = predict_nlp.predict_complaint(request.complaint_text)

        # 3. Identity score
        i_score = predict_cv.predict_kyc(request.selfie_path, request.id_path)

        # 4. Fusion
        final_score, decision = fusion_engine.final_fraud_score(t_score, c_score, i_score)

        return {
            "status": "success",
            "results": {
                "transaction_score": round(t_score, 4),
                "complaint_score": round(c_score, 4),
                "identity_score": round(i_score, 4),
                "final_score": round(final_score, 4),
                "decision": decision
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AI Fraud Detection API is running"}
