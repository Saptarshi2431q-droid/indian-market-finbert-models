"""
Layer 12: Production Inference Engine (Deep Archive Scaling)
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Execute batched inference on 40,000 timestamped headlines using RTX Tensor Cores.
"""

import os
# Structural override for concurrent OpenMP execution ("DLL Hell" bypass)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

print("\n--- STARTING DEEP ARCHIVE PRODUCTION INFERENCE ---", flush=True)

print("1. Loading historical dataset...", flush=True)

# Pure Python parsing to safely ingest 40,000 un-sanitized real-world text strings
try:
    # UPDATED: Pointing to the new deep archive we just scraped
    df = pd.read_csv("timestamped_historical_headlines.csv", engine='python', encoding='utf-8', on_bad_lines='skip')
    total_headlines = len(df)
    print(f"[SUCCESS] Loaded {total_headlines} chronological headlines for processing.", flush=True)
except Exception as e:
    print(f"[FATAL ERROR DURING CSV LOAD]: {e}")
    exit()

print("\n2. Waking up the RTX 4050 and PyTorch (this takes a moment)...", flush=True)
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

model_path = "./optimized_indian_finbert_final"
print(f"Loading custom fine-tuned neural network from: {model_path}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = {0: "Bullish", 1: "Bearish", 2: "Neutral"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"3. Compute Hardware Verified: {device.type.upper()}", flush=True)

# We keep the batch size at 32 to safely maximize your 6GB VRAM
batch_size = 32 
sentiments = []
confidences = []

print("\n4. Executing Native Production Inference Pipeline...", flush=True)
start_time = time.time()

with torch.no_grad():
    for i in range(0, total_headlines, batch_size):
        batch_texts = [str(text) for text in df['Headline'].iloc[i : i + batch_size].tolist()]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        for j in range(len(batch_texts)):
            pred_idx = np.argmax(probs[j])
            
            final_sentiment = label_map[pred_idx]
            confidence = probs[j][pred_idx]
            
            sentiments.append(final_sentiment)
            confidences.append(confidence)
            
        # Print a progress update every ~3,200 headlines
        if i % 3200 == 0 and i > 0:
            print(f"Processed {i}/{total_headlines} headlines...", flush=True)

end_time = time.time()
df['Predicted_Sentiment'] = sentiments
df['Confidence_Score'] = confidences

print("\n" + "="*50, flush=True)
print("   DEEP ARCHIVE INFERENCE EXECUTION METRICS", flush=True)
print("="*50, flush=True)
print(f"Total Time Taken   : {end_time - start_time:.2f} seconds", flush=True)
print(f"Processing Speed   : {total_headlines / (end_time - start_time):.2f} headlines/sec", flush=True)
print("="*50, flush=True)

# UPDATED: Saving to a new, distinct filename for the Multimodal merge
output_filename = "deep_archive_classified_headlines.csv"
df.to_csv(output_filename, index=False)
print(f"\n[SUCCESS] Production dataset saved as '{output_filename}'", flush=True)