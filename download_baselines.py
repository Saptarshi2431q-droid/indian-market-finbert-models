"""
Layer 12: Hugging Face Baseline Acquisition
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Download and cache competitor models for Champion vs. Challenger testing.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def fetch_and_save_hf_model(repo_id, local_folder_name):
    print("==================================================")
    print(f"  INITIATING DOWNLOAD: {repo_id}")
    print("==================================================")
    
    # Create the directory if it doesn't exist
    os.makedirs(local_folder_name, exist_ok=True)
    
    try:
        # 1. Download the Tokenizer (The Dictionary)
        print("[1/2] Downloading Tokenizer architecture...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        # 2. Download the Neural Network Weights (The Brain)
        print("[2/2] Downloading 400MB+ Model weights. This may take a minute...")
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        
        # 3. Save permanently to your local hard drive
        print(f"Saving assets to local directory: {local_folder_name}...")
        tokenizer.save_pretrained(local_folder_name)
        model.save_pretrained(local_folder_name)
        
        print(f"[SUCCESS] {repo_id} successfully locked in the vault.\n")
        
    except Exception as e:
        print(f"[FATAL ERROR] Could not download {repo_id}. Check your spelling or internet connection.")
        print(f"Error Details: {e}\n")

def main():
    # Model 1: The Western FinBERT Baseline
    western_repo = "ProsusAI/finbert"
    western_folder = "./models/western_finbert_baseline"
    
    # Model 2: The Indian FinBERT Competitor
    # REPLACE "placeholder-user/indian-finbert" WITH THE ACTUAL HF REPO ID YOU FIND
    indian_repo = "placeholder-user/indian-finbert" 
    indian_folder = "./models/hf_indian_finbert_baseline"
    
    fetch_and_save_hf_model(western_repo, western_folder)
    
    # Uncomment the line below once you find the exact Indian FinBERT Repo ID
    # fetch_and_save_hf_model(indian_repo, indian_folder)

if __name__ == "__main__":
    main()