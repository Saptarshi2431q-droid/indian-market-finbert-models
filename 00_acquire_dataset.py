"""
Step 0: Institutional Data Acquisition
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Download a clean, human-annotated 11.9k financial dataset to eliminate pseudo-labeling bias.
"""
import pandas as pd
from datasets import load_dataset

def main():
    print("==================================================")
    print("  FETCHING UNBIASED 11.9K FINANCIAL DATASET")
    print("==================================================")

    try:
        # 1. Download the human-annotated dataset from Hugging Face
        print("Downloading dataset from Hugging Face servers (this may take a moment)...")
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
        
        # 2. Convert Hugging Face format to Pandas DataFrames
        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        
        # Combine them to get the full 11.9k rows
        full_df = pd.concat([train_df, valid_df], ignore_index=True)
        print(f"Successfully loaded {len(full_df)} raw headlines.")

        # 3. Format the data to match your AI4Invest Pipeline exactly
        # This dataset uses: 0 = Bearish, 1 = Bullish, 2 = Neutral
        label_mapping = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
        
        full_df['Predicted_Sentiment'] = full_df['label'].map(label_mapping)
        full_df.rename(columns={'text': 'Headline'}, inplace=True)
        
        # Keep only the columns we need for the pipeline
        final_df = full_df[['Headline', 'Predicted_Sentiment']]
        
        # 4. Save to CSV
        file_name = "unbiased_financial_headlines.csv"
        final_df.to_csv(file_name, index=False)
        print(f"\n[SUCCESS] Dataset formatted and saved as '{file_name}'.")
        print("This file contains ZERO pseudo-labeling bias and is ready for the Gladiatorial Arena.")

    except Exception as e:
        print(f"\n[FATAL ERROR] Could not fetch the dataset. {e}")

if __name__ == "__main__":
    main()