"""
Bulk Inference Engine: Real-World Headline Evaluation
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Run optimized domain-adapted classification across scraped headlines.
"""

import torch
import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

def main():
    print("Loading historical dataset...")
    try:
        df = pd.read_csv("historical_5k_headlines.csv")
    except FileNotFoundError:
        print("[ERROR] Could not find the dataset. Run the scraper first!")
        return
        
    total_headlines = len(df)
    print(f"Loaded {total_headlines} headlines for processing.")

    # FIX 1: We use AutoModelForSequenceClassification to keep the classification head intact!
    print("Initializing Indian FinBERT Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    # FinBERT label map: 0 = Positive (Bullish), 1 = Negative (Bearish), 2 = Neutral
    label_map = {0: "Bullish", 1: "Bearish", 2: "Neutral"}

    # Hardware Optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Compute Hardware: {device.type.upper()}")

    batch_size = 32 
    sentiments = []
    confidences = []
    
    print("\nExecuting Batched Inference Pipeline...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, total_headlines, batch_size):
            # Grab the batch of headlines. Convert all to strings to prevent float errors.
            batch_texts = [str(text) for text in df['Headline'].iloc[i : i + batch_size].tolist()]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            
            # FIX 2: Apply Softmax to get the true mathematical probabilities of the classes
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                # Extract the base FinBERT prediction
                pred_idx = np.argmax(probs[j])
                base_sentiment = label_map[pred_idx]
                confidence = probs[j][pred_idx]
                
                # FIX 3: Apply our Phase 4 Indian Domain Adaptation Override
                # We manually force the model to classify localized Dalal Street risk terms as Bearish
                text_lower = text.lower()
                indian_risk_terms = ['crr', 'repo rate', 'inflation', 'deficit', 'npa', 'sebi warning', 'penalty']
                
                if any(kw in text_lower for kw in indian_risk_terms):
                    final_sentiment = "Bearish"
                else:
                    final_sentiment = base_sentiment
                    
                sentiments.append(final_sentiment)
                confidences.append(confidence)
                
            if i % 320 == 0 and i > 0:
                print(f"Processed {i}/{total_headlines} headlines...")

    end_time = time.time()
    df['Predicted_Sentiment'] = sentiments
    df['Confidence_Score'] = confidences
    
    print("\n" + "="*50)
    print("   BULK INFERENCE EXECUTION METRICS")
    print("="*50)
    print(f"Total Time Taken   : {end_time - start_time:.2f} seconds")
    print(f"Processing Speed   : {total_headlines / (end_time - start_time):.2f} headlines/sec")
    print("="*50)

    # Calculate overall market sentiment distribution
    bullish_count = len(df[df['Predicted_Sentiment'] == 'Bullish'])
    bearish_count = len(df[df['Predicted_Sentiment'] == 'Bearish'])
    neutral_count = len(df[df['Predicted_Sentiment'] == 'Neutral'])
    
    print("\n--- Market Sentiment Distribution ---")
    print(f"Bullish Headlines : {bullish_count} ({(bullish_count/total_headlines)*100:.1f}%)")
    print(f"Bearish Headlines : {bearish_count} ({(bearish_count/total_headlines)*100:.1f}%)")
    print(f"Neutral Headlines : {neutral_count} ({(neutral_count/total_headlines)*100:.1f}%)")

    # Save the final product
    output_filename = "classified_historical_headlines.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Final classified dataset saved as '{output_filename}'")
    print("Ready for Durga Bhaiyaa's review!")

if __name__ == "__main__":
    main()