import sys
import os
import importlib.util

# Function to load modules from paths with numeric prefixes
def load_module_from_path(module_name, file_path):
    module_dir = os.path.dirname(file_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Resolve absolute paths for the modules
base_path = os.path.dirname(os.path.abspath(__file__))

# Load models dynamically
predict_dl = load_module_from_path("predict_dl", os.path.join(base_path, "1_transactions_DL", "predict.py"))
predict_nlp = load_module_from_path("predict_nlp", os.path.join(base_path, "2_complaints_NLP", "predict.py"))
predict_cv = load_module_from_path("predict_cv", os.path.join(base_path, "3_kyc_CV", "predict.py"))
fusion_engine = load_module_from_path("fusion_engine", os.path.join(base_path, "4_fusion_engine", "fraud_score.py"))

def run_full_system(transaction_data, complaint_text, selfie_path, id_path):

    print("\n[1/4] Running Transaction Model...")
    transaction_score = predict_dl.predict_transaction(transaction_data)
    print(f"Transaction Score: {round(transaction_score, 4)}")

    print("\n[2/4] Running Complaint NLP Model...")
    complaint_score = predict_nlp.predict_complaint(complaint_text)
    print(f"Complaint Score: {round(complaint_score, 4)}")

    print("\n[3/4] Running KYC Model...")
    identity_score = predict_cv.predict_kyc(selfie_path, id_path)
    print(f"Identity Score: {round(identity_score, 4)}")

    print("\n[4/4] Running Fusion Engine...")
    final_score, decision = fusion_engine.final_fraud_score(
        transaction_score,
        complaint_score,
        identity_score
    )

    return {
        "transaction_score": transaction_score,
        "complaint_score": complaint_score,
        "identity_score": identity_score,
        "final_score": final_score,
        "decision": decision
    }


if __name__ == "__main__":
    print("=== AI Fraud Detection System Integrated Pipeline ===")

    # Correct sample data format for the transaction model
    sample_transaction = {
        "amount": 2500,
        "type": "TRANSFER",
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 7500
    }
    
    complaint_text = "I think there is an unauthorized transaction on my account."
    selfie = "selfie.jpg"
    id_card = "id.jpg"

    result = run_full_system(
        sample_transaction,
        complaint_text,
        selfie,
        id_card
    )

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:18}: {round(value, 4)}")
        else:
            print(f"{key:18}: {value}")
    print("="*40)
