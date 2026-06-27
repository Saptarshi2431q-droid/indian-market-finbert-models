from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def fetch_model(repo_id, local_folder):
    print(f"\n--- DOWNLOADING: {repo_id} ---")
    os.makedirs(local_folder, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        tokenizer.save_pretrained(local_folder)
        model.save_pretrained(local_folder)
        print(f"[SUCCESS] Saved to {local_folder}")
    except Exception as e:
        print(f"[FATAL ERROR] Could not download {repo_id}. {e}")

def main():
    print("INITIALIZING HUGGING FACE DOWNLOADER...")
    fetch_model("ProsusAI/finbert", "./models/western_finbert_baseline")
    
    # NEW COMPETITOR INJECTED
    fetch_model("Vansh180/FinBERT-India-v1", "./models/hf_indian_finbert_baseline")

if __name__ == "__main__":
    main()