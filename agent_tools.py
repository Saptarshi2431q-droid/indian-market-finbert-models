"""
Project: AI4Invest MLOps Agent Framework
Module: Tool Execution Layer (Sanitized)
Author: Saptarshi Dutta (Rishi)
"""

import json
import yfinance as yf
from transformers import pipeline

print("[INIT] Bootstrapping proprietary AI4Invest FinBERT NLP Pipeline...")
try:
    sentiment_pipeline = pipeline("text-classification", model="sapt3009/AI4Invest-Indian-FinBERT")
    print("[SUCCESS] Cloud weights successfully cached into active execution space.")
except Exception as e:
    print(f"[ERROR] Failed to download cloud weights: {e}")
    sentiment_pipeline = None

def get_live_price(ticker: str) -> str:
    """Fetches the latest closing market price from the NSE."""
    try:
        stock = yf.Ticker(f"{ticker.strip().upper()}.NS")
        hist = stock.history(period="1d")
        if hist.empty:
            return json.dumps({"ticker": ticker, "current_price": "Data Unavailable"})
        
        return json.dumps({"ticker": ticker.upper(), "current_price": round(float(hist['Close'].iloc[-1]), 2)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_financial_news(ticker: str) -> str:
    """
    [TASK 1 BASELINE HACK]
    Bypasses Yahoo Finance and aggressively loads a massive JSON payload 
    to test the Maximum Effective Context Window (MECW) and trigger rate limits.
    """
    import os
    print(f"[SYSTEM WARNING] Fetching bloated 100-article payload for {ticker}...")
    
    file_path = "100_articles_dump.json"
    
    if not os.path.exists(file_path):
        return json.dumps(["[ERROR] 100_articles_dump.json not found."])
        
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            massive_payload = json.load(file)
            
        # We are intentionally NOT slicing this. We want the full payload to hit the LLM.
        return json.dumps(massive_payload)
        
    except Exception as e:
        return json.dumps([f"Error reading local dump: {str(e)}"])

def analyze_stock_sentiment(headlines_json: str) -> str:
    """
    Safely parses whatever the LLM passes and runs the FinBERT classification.
    """
    if not sentiment_pipeline:
        return json.dumps({"error": "FinBERT offline."})
    
    try:
        # Bulletproof JSON parsing
        try:
            headlines = json.loads(headlines_json)
        except:
            headlines = [headlines_json]

        # Handle edge cases where LLM passes a dict instead of a list
        if isinstance(headlines, dict):
            headlines = headlines.get("headlines", list(headlines.values()))
        elif not isinstance(headlines, list):
            headlines = [str(headlines)]
            
        # Filter out empty data
        clean_text = [str(h) for h in headlines if str(h).strip()]
        
        if not clean_text:
            return json.dumps({"error": "No readable text provided."})
        
        inference_outputs = sentiment_pipeline(clean_text)
        
        bullish = sum(1 for output in inference_outputs if output['label'] == 'LABEL_0')
        bearish = sum(1 for output in inference_outputs if output['label'] == 'LABEL_1')
        neutral = sum(1 for output in inference_outputs if output['label'] == 'LABEL_2')
        
        return json.dumps({
            "total_analyzed": len(clean_text),
            "bullish_signals": bullish,
            "bearish_signals": bearish,
            "neutral_signals": neutral
        })
    except Exception as e:
        return json.dumps({"error": f"Sentiment analysis failed: {str(e)}"})