import sys
import os
import importlib.util

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from random import random
import torch
import joblib
import pandas as pd

def load_module_from_path(module_name, file_path):
    module_dir = os.path.dirname(file_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_identity_score():
    """Calculates identity trust score using the CV model."""
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../3_kyc_CV'))
        predict_cv = load_module_from_path("predict_cv", os.path.join(base_path, "predict.py"))
        return predict_cv.predict_kyc("selfie.jpg", "id.jpg")
    except Exception as e:
        print(f"Error loading identity model: {e}. Using default.")
        return 0.64

def get_transaction_score():
    """Calculates transaction fraud score using the DL model."""
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../1_transactions_DL'))
        
        # Add 1_transactions_DL to sys.path so it can find train_model
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
            
        # Load the module dynamically
        predict_dl = load_module_from_path("predict_dl", os.path.join(base_path, "predict.py"))
        return predict_dl.predict_transaction()
    except Exception as e:
        print(f"Error loading transaction model: {e}. Using default.")
        return 0.44

def get_complaint_score():
    """Calculates complaint fraud score using the NLP model."""
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../2_complaints_NLP'))
        
        # Add 2_complaints_NLP to sys.path
        if base_path not in sys.path:
            sys.path.insert(0, base_path)

        predict_nlp = load_module_from_path("predict_nlp", os.path.join(base_path, "predict.py"))
        
        test_text = "Money was deducted without my permission"
        return predict_nlp.predict_complaint(test_text)
    except Exception as e:
        print(f"Error loading complaint model: {e}. Using default.")
        return 0.72

def final_fraud_score(transaction_score, complaint_score, identity_score):
    final_score = (
        0.5 * transaction_score +
        0.3 * complaint_score +
        0.2 * (1 - identity_score)
    )

    if final_score > 0.7:
        decision = "FRAUD"
    else:
        decision = "LEGIT"

    return final_score, decision


if __name__ == "__main__":
    print("--- AI Fraud Detection Fusion Engine ---")
    
    # 1. Get score from DL Transaction Model
    t_score = get_transaction_score()
    print(f"Transaction DL Score: {round(t_score, 3)}")

    # 2. Get score from NLP Complaint Model
    c_score = get_complaint_score()
    print(f"Complaint NLP Score: {round(c_score, 3)}")

    # 3. Get score from CV Identity Model
    i_score = get_identity_score()
    print(f"Identity CV Score: {round(i_score, 3)}")

    # 4. Fusion
    score, decision = final_fraud_score(t_score, c_score, i_score)

    print("\n" + "="*40)
    print("FINAL RISK SCORE  :", round(score, 3))
    print("FINAL DECISION    :", decision)
    print("="*40)
