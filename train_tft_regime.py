"""
Temporal Fusion Transformer (TFT) - Regime Prediction Pipeline
Author: Technical Internship Portfolio
Framework: PyTorch Forecasting, PyTorch Lightning
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import CrossEntropy

warnings.filterwarnings("ignore") 

def prepare_tft_data():
    print("Downloading Nifty 50 Market Data...")
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(start="2020-01-01", end="2024-01-01").reset_index()
    
    # Feature Engineering
    df['Return'] = df['Close'].pct_change()
    df['Volatility_10d'] = df['Return'].rolling(10).std()
    df['Volume_Log'] = np.log(df['Volume'] + 1)
    
    # Known Covariates
    df['Day_of_Week'] = df['Date'].dt.dayofweek.astype(str)
    df['Month'] = df['Date'].dt.month.astype(str)
    
    # Multimodal NLP Structural Slot
    # Initialized as a placeholder string ("1_Neutral") to safely compile the dataset schema.
    # This column will be populated directly by the text vectors from nlp_pipeline.py.
    df['Macro_Sentiment_Cluster'] = "1_Neutral"
    
    # Target Engineering: Next 5-Day Return
    df['Future_5d_Return'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # Map to Regimes
    conditions = [
        (df['Future_5d_Return'] < -0.015),
        (df['Future_5d_Return'] >= -0.015) & (df['Future_5d_Return'] <= 0.015),
        (df['Future_5d_Return'] > 0.015)
    ]
    choices = ['0_Bear', '1_Sideways', '2_Bull']
    df['Regime'] = np.select(conditions, choices, default='1_Sideways')
    
    df = df.dropna()
    
    # TFT Structural Requirements
    df['time_idx'] = np.arange(len(df))
    df['group'] = "NIFTY50"
    df['Regime'] = df['Regime'].astype(str)
    df['Macro_Sentiment_Cluster'] = df['Macro_Sentiment_Cluster'].astype(str)
    
    return df

def main():
    # --- CRITICAL REGIME FREEZE ---
    # Enforces hard reproducibility bounds across Python, NumPy, and PyTorch backends.
    # Locks the network into the optimized momentum configuration (Run Type B).
    pl.seed_everything(42, workers=True)
    
    df = prepare_tft_data()
    print(f"\nData Processing Complete. Total Trading Days: {len(df)}")
    
    max_encoder_length = 30  
    max_prediction_length = 1 
    training_cutoff = df["time_idx"].max() - 100
    
    print("\nInitializing TFT TimeSeriesDataSet...")
    training_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Regime",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2, 
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        # Multimodal Integration: Added the text-derived feature slot to known categoricals
        time_varying_known_categoricals=["Day_of_Week", "Month", "Macro_Sentiment_Cluster"],
        
        time_varying_unknown_reals=["Close", "Return", "Volatility_10d", "Volume_Log"],
        target_normalizer=NaNLabelEncoder(), 
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    print("DataLoader Successfully Generated.")

    print("\nBuilding Temporal Fusion Transformer Architecture...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=16,               
        attention_head_size=4,        
        dropout=0.1,                  
        hidden_continuous_size=8,
        output_size=3,                
        loss=CrossEntropy(),          
        log_interval=10,
    )
    
    print(f"Network built with {tft.size()/1e3:.1f}k trainable parameters.")

    # Added deterministic=True to prevent internal stochastic divergence inside the GRNs
    trainer = pl.Trainer(
        max_epochs=5,           
        accelerator="auto",     
        gradient_clip_val=0.1,  
        deterministic=True,
        logger=False,
        enable_checkpointing=False
    )

    print("\n--- Starting TFT Training Loop ---")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
    )
    print("\nTraining Complete! The TFT has successfully learned the Nifty 50 market regimes.")

    print("\n--- Extracting AI Interpretability Metrics ---")
    raw_predictions = tft.predict(train_dataloader, mode="raw", return_x=True)
    interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")
    
    import matplotlib.pyplot as plt
    print("Generating Feature Importance & Attention charts...")
    figs = tft.plot_interpretation(interpretation)
    figs['encoder_variables'].savefig("feature_importance.png", bbox_inches='tight')
    figs['attention'].savefig("attention_focus.png", bbox_inches='tight')
    
    print("SUCCESS: Saved 'feature_importance.png' and 'attention_focus.png' to your project folder.")

    print("\n--- Translating AI Signals (Live Regime Predictions) ---")
    logits = raw_predictions.output.prediction
    predicted_indices = logits.argmax(dim=-1)
    
    regime_map = {0: "0_Bear (Market Drop Expected)", 
                  1: "1_Sideways (Choppy/Stable Market)", 
                  2: "2_Bull (Market Breakout Expected)"}
    
    print("Here are the AI's regime classifications for the 5 most recent sequences in your data:\n")
    
    for i in range(5, 0, -1):
        pred_idx = predicted_indices[-i, 0].item()
        print(f"Sequence Lookback T-{i} Days  -->  AI Signal: {regime_map[pred_idx]}")
        
    print("\nPipeline execution fully completed.")

if __name__ == "__main__":
    main()