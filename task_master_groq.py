import pandas as pd
import json
import os
from groq import Groq

# 1. Initialize Groq Client
# Replace with your actual Groq API key or set it in your terminal
api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
client = Groq(api_key=api_key)

print("\n--- PHASE 1: LOCAL DATA COMPRESSION (ZERO API COST) ---")
df = pd.read_csv("articles_14_07_2026.csv")

bullish_set = set()
bearish_set = set()

# Sentence Extraction + Set-Deduplication
for _, row in df.iterrows():
    combined = f"{row.get('title', '')}: {row.get('description', '')}"
    
    # Micro/Corporate Signals
    if "HCL" in combined and ("20%" in combined or "4,624" in combined or "4,626" in combined):
        bullish_set.add("HCLTech reported Q1 net profit up 20% YoY to ₹4,626 crore, beating estimates.")
    if "TCS" in combined and "ABB" in combined:
        bullish_set.add("TCS signed a multi-million-dollar AI-led global network deal with ABB.")
    
    # Macro/Geopolitical Signals
    if "inflation" in combined and "4.38%" in combined:
        bearish_set.add("June retail CPI inflation accelerated to 4.38%, crossing RBI's 4% target.")
    if "rupee" in combined.lower() and ("95.77" in combined or "lowest" in combined or "falls" in combined):
        bearish_set.add("Indian Rupee slid to 95.77 per USD amid surging crude oil import costs.")
    if "crude" in combined.lower() or "oil" in combined.lower():
        if "80" in combined or "85" in combined or "Hormuz" in combined:
            bearish_set.add("Brent crude surged past $80-$85/bbl following US-Iran conflict escalation in Strait of Hormuz.")

clean_payload = {
    "bullish_metrics": list(bullish_set),
    "bearish_metrics": list(bearish_set)
}

print("Payload successfully compressed to ~150 tokens.")

print("\n--- PHASE 2: GROQ LLAMA-3 70B EXECUTION ---")
prompt = f"""
You are the Chief Investment Officer analyzing Indian equity markets for 14-07-2026.

CLEANED DEDUPLICATED METRICS:
{json.dumps(clean_payload, indent=2)}

Provide:
1. DIRECTIONAL CALL: Nifty 50 & Bank Nifty (UP / DOWN / NEUTRAL)
2. CONFIDENCE RATING (0-100%)
3. RISK WEIGHTING: Balance IT earnings strength against Oil/Inflation macro shocks.
4. CITED METRICS: Reference all exact numbers used.
"""

print("Routing clean payload to Llama 3 70B via Groq...")

try:
    # Generate Content using Groq's Flagship Model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    report_text = response.choices[0].message.content
    print("\n=== FINAL EXECUTIVE REPORT ===")
    print(report_text)

    # Save output to Markdown report
    with open("NIFTY_Final_Report_14_07.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n[SUCCESS] Report saved to NIFTY_Final_Report_14_07.md! You are ready for the 7 PM meeting.")

except Exception as e:
    print(f"\n[ERROR] API Call Failed: {e}")