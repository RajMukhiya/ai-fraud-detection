import torch
import joblib
import pandas as pd
import os
from train_model import FraudModel

# Use absolute paths for loading files
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

le = joblib.load(os.path.join(BASE_PATH, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))

def predict_transaction(data_dict=None):
    if data_dict is None:
        data_dict = {
            "amount": 1500,
            "type": "TRANSFER",
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 3500
        }
    
    sample = pd.DataFrame([data_dict])
    
    sample["type"] = le.transform(sample["type"])
    sample[["amount","oldbalanceOrg","newbalanceOrig"]] = scaler.transform(
        sample[["amount","oldbalanceOrg","newbalanceOrig"]]
    )
    
    X = torch.tensor(sample.values, dtype=torch.float32)
    
    model = FraudModel(X.shape[1])
    model.load_state_dict(torch.load(os.path.join(BASE_PATH, "transaction_model.pt")))
    model.eval()
    
    with torch.no_grad():
        score = model(X).item()
    return score

if __name__ == "__main__":
    score = predict_transaction()
    print("Transaction Fraud Score:", round(score, 2))
