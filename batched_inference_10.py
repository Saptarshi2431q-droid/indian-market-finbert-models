"""
Indian FinBERT Inference Profiler: Sequential vs Batched Processing
Author: Technical Internship Portfolio
Objective: Quantify the hardware acceleration achieved via Static Batching.
"""

import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")

def generate_100_headlines():
    # Helper to instantly generate 100 dummy Indian market headlines for benchmarking
    return ["RBI unexpectedly hikes CRR by 50 bps to drain liquidity."] * 25 + \
           ["TCS beats street estimates with massive margin expansion."] * 25 + \
           ["Infosys misses Q3 revenue guidance, shares plummet."] * 25 + \
           ["HDFC twin merger finalized, creating global banking giant."] * 25

def main():
    print("Loading Indian FinBERT Architecture...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    headlines = generate_100_headlines()
    total_headlines = len(headlines)
    
    # Enable CPU/GPU optimization flags if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\n==================================================")
    print("   RUN 1: SEQUENTIAL INFERENCE (BATCH SIZE = 1)   ")
    print("==================================================")
    
    start_time_seq = time.time()
    
    with torch.no_grad():
        for text in headlines:
            # The bottleneck: Sending one sequence to the processor at a time
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            _ = outputs.logits
            
    end_time_seq = time.time()
    seq_duration = end_time_seq - start_time_seq

    print("\n==================================================")
    print("    RUN 2: STATIC BATCHING (BATCH SIZE = 10)      ")
    print("==================================================")
    
    batch_size = 10
    start_time_batched = time.time()
    
    with torch.no_grad():
        # Step through the dataset in chunks of 10
        for i in range(0, total_headlines, batch_size):
            batch_texts = headlines[i : i + batch_size]
            
            # The Optimization: Tokenizer pads all 10 sentences to the same length
            # and stacks them into a single mathematical tensor [10, Sequence_Length]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # The GPU computes all 10 forward passes simultaneously
            outputs = model(**inputs)
            _ = outputs.logits
            
    end_time_batched = time.time()
    batched_duration = end_time_batched - start_time_batched

    print("\n--- INFERENCE HARDWARE PROFILING RESULTS ---")
    print(f"Total Headlines Processed : {total_headlines}")
    print(f"Sequential Time (B=1)     : {seq_duration:.4f} seconds")
    print(f"Batched Time (B=10)       : {batched_duration:.4f} seconds")
    
    speedup = seq_duration / batched_duration
    print(f"\n[METRIC] Static Batching achieved a {speedup:.2f}x hardware acceleration multiplier.")

if __name__ == "__main__":
    main()