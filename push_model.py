from huggingface_hub import HfApi, login

print("Authenticating with Hugging Face...")
login()

api = HfApi()

LOCAL_MODEL_DIR = "./models/optimized_indian_finbert_final"
REPO_ID = "sapt3009/AI4Invest-Indian-FinBERT" 

print(f"Initiating uplink: Transferring weights to {REPO_ID}...")

api.upload_folder(
    folder_path=LOCAL_MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Deploying Phase 2 SOTA weights (MCC: 0.7478, Accuracy: 86.8%)"
)

print("[SUCCESS] Model deployed.")