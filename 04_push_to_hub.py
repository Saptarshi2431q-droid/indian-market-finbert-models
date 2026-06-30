"""
Step 4: Hugging Face Cloud Deployment
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Push the fine-tuned custom model securely to the Hugging Face Hub.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import os

def main():
    print("==================================================")
    print("  INITIATING HUGGING FACE CLOUD DEPLOYMENT")
    print("==================================================")
    
    # 1. Authenticate with Hugging Face
    # Paste your Write Token inside the quotes below
    HF_TOKEN = "YOUR_HF_TOKEN_HERE" 
    
    try:
        login(token=HF_TOKEN)
        print("\n[SUCCESS] Authenticated with Hugging Face.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Authentication failed. Check your token. {e}")
        return

    # 2. Define the exact path to your custom AI4Invest model
    local_model_path = "./models/optimized_indian_finbert_final"
    
    # 3. Define what you want your model to be named on the internet
    # Change 'your_hf_username' to your actual Hugging Face username!
    hf_username = "sapt3009"
    repo_name = "AI4Invest-Indian-FinBERT" 
    full_repo_id = f"{hf_username}/{repo_name}"

    print(f"\nLoading local weights from: {local_model_path}")
    print(f"Target Destination: https://huggingface.co/{full_repo_id}")
    
    try:
        # Load the brain and the dictionary into RAM
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        
        # 4. Push to the Cloud
        print("\nUploading Tokenizer... (This is fast)")
        tokenizer.push_to_hub(full_repo_id, private=False) # private=False makes it open-source!
        
        print("Uploading 400MB+ Neural Network Weights... (This may take a few minutes)")
        model.push_to_hub(full_repo_id, private=False)
        
        print("\n==================================================")
        print(f" [DEPLOYMENT COMPLETE] Model is live!")
        print(f" Link: https://huggingface.co/{full_repo_id}")
        print("==================================================")

    except Exception as e:
        print(f"\n[ERROR] Deployment failed. {e}")

if __name__ == "__main__":
    main()