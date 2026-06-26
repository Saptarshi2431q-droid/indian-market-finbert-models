"""
Layer 13: True Multimodal Data Fusion (Deep Archive)
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Hard-merge true chronological NLP Sentiment with 5-year Nifty 50 Prices.
"""

import pandas as pd
import yfinance as yf
import numpy as np

def main():
    print("--- STARTING TRUE CHRONOLOGICAL MULTIMODAL FUSION ---")

    # 1. Load the Deep Archive Sentiment Data
    print("1. Loading 40,000-row chronological sentiment data...")
    sentiment_df = pd.read_csv("deep_archive_classified_headlines.csv")
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date

    # 2. Download 5 Years of Nifty 50 Historical Data
    print("2. Pinging Yahoo Finance API for 5-Year Nifty 50 (^NSEI) history...")
    nifty_data = yf.download("^NSEI", period="5y", interval="1d")
    
    # Flatten the multi-index columns from yfinance
    nifty_data.columns = nifty_data.columns.get_level_values(0)
    nifty_data = nifty_data.reset_index()
    nifty_data['Date'] = pd.to_datetime(nifty_data['Date']).dt.date

    # 3. True Mathematical Aggregation (The No-Compromise Fix)
    print("3. Aggregating daily macroeconomic sentiment vectors...")
    
    # Group the 40k headlines strictly by their real publication dates
    grouped = sentiment_df.groupby('Date')['Predicted_Sentiment'].value_counts().unstack(fill_value=0)
    
    # Ensure columns exist even if some days had zero bullish/bearish news
    for col in ['Bullish', 'Bearish', 'Neutral']:
        if col not in grouped.columns:
            grouped[col] = 0
            
    grouped['Total'] = grouped['Bullish'] + grouped['Bearish'] + grouped['Neutral']
    
    # Calculate the Net Sentiment Score [-1.0 to 1.0]
    grouped['Daily_NLP_Sentiment'] = (grouped['Bullish'] - grouped['Bearish']) / grouped['Total']
    
    # Isolate the Date (index) and the final score
    sentiment_scores = grouped[['Daily_NLP_Sentiment']].reset_index()

    # 4. The Hard Temporal Merge
    print("4. Executing strict temporal merge (Left Join on Trading Days)...")
    # We merge sentiment onto the Nifty data. If a trading day had zero news, it defaults to Neutral (0.0)
    master_df = pd.merge(nifty_data, sentiment_scores, on='Date', how='left')
    master_df['Daily_NLP_Sentiment'] = master_df['Daily_NLP_Sentiment'].fillna(0.0)

    # 5. Feature Engineering (Technical Context)
    print("5. Engineering Technical Indicators...")
    master_df['SMA_14'] = master_df['Close'].rolling(window=14).mean()
    master_df['Daily_Volatility'] = master_df['High'] - master_df['Low']
    
    # Drop rows with incomplete 14-day history to protect the PyTorch tensors
    master_df = master_df.dropna()

    output_filename = "deep_archive_multimodal_master.csv"
    master_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*50)
    print("   TRUE MULTIMODAL FUSION COMPLETE")
    print("="*50)
    print(f"Total Trading Days Processed : {len(master_df)}")
    print(f"Features Engineered          : Price, Volatility, SMA, NLP Sentiment")
    print(f"Dataset securely saved as    : '{output_filename}'")
    print("="*50)
    print("Ready for the Ultimate Temporal Fusion Transformer!")

if __name__ == "__main__":
    main()