"""
Layer 14: Temporal Fusion Transformer Training & Prediction (Deep Archive)
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Predict Nifty 50 price action using a 5-Year Chronological Multimodal Dataset.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import warnings

warnings.filterwarnings("ignore")

# THE TENSOR CORE OPTIMIZATION: Unlocks the RTX 4050's maximum speed
torch.set_float32_matmul_precision('medium')

print("\n--- INITIATING TEMPORAL FUSION TRANSFORMER (DEEP ARCHIVE) ---", flush=True)

# UPDATED: We are now loading the 5-Year True Chronological Dataset
df = pd.read_csv("deep_archive_multimodal_master.csv")
df["time_idx"] = np.arange(len(df))
df["group"] = "Nifty50"

numeric_cols = ["Close", "Volume", "Daily_NLP_Sentiment", "SMA_14", "Daily_Volatility"]
for col in numeric_cols:
    df[col] = df[col].astype(float)

max_prediction_length = 5  
max_encoder_length = 30    
training_cutoff = df["time_idx"].max() - max_prediction_length

training_dataset = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    group_ids=["group"],
    min_encoder_length=max_encoder_length // 2, 
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Close", "Volume", "SMA_14", "Daily_Volatility", "Daily_NLP_Sentiment"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, df, predict=True, stop_randomization=True)

train_dataloader = training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)
val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

tft_model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.01,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(), 
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto", 
    devices="auto",
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False
)

trainer.fit(tft_model, train_dataloaders=train_dataloader)

# =====================================================================
# THE CRYSTAL BALL: PREDICTING THE FUTURE
# =====================================================================
print("\n" + "="*50)
print("   GENERATING 5-DAY MARKET FORECAST")
print("="*50)
print("The AI is blindly predicting the final 5 days of the dataset")
print("using 5 Years of true chronological Nifty 50 history and News Sentiment...\n")

actual_prices = df['Close'].tail(5).values

raw_predictions = tft_model.predict(val_dataloader)
ai_forecast = raw_predictions[0].cpu().numpy()

for i in range(5):
    error_margin = abs(ai_forecast[i] - actual_prices[i])
    print(f"Day {i+1} | AI Forecast: {ai_forecast[i]:.2f} INR | Actual Market: {actual_prices[i]:.2f} INR | Error: {error_margin:.2f}")

print("\n[PIPELINE COMPLETE] You have built the ultimate Multimodal Quantitative forecaster!")
print("==================================================")