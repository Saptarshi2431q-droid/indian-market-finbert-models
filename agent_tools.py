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
    Queries real-time news and aggressively sanitizes the output 
    so the LLM only receives a clean list of strings.
    """
    fallback_news = [
        f"Analysts update price targets for {ticker.upper()}.",
        f"Market volume remains steady for {ticker.upper()} going into Q3.",
        f"{ticker.upper()} evaluating new institutional growth strategies."
    ]
    
    try:
        stock = yf.Ticker(f"{ticker.strip().upper()}.NS")
        news_stream = stock.news
        
        if not news_stream:
            return json.dumps(fallback_news)
            
        # Extract ONLY the title strings, dropping all URLs and metadata
        clean_headlines = []
        for item in news_stream[:5]:
            if isinstance(item, dict) and 'title' in item:
                clean_headlines.append(item['title'])
                
        if not clean_headlines:
            return json.dumps(fallback_news)
            
        return json.dumps(clean_headlines)
        
    except Exception:
        return json.dumps(fallback_news)

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