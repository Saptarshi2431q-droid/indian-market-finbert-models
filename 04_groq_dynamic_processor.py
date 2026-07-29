import pandas as pd
import json
import os
from groq import Groq

# 1. Initialize Groq Client
api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
client = Groq(api_key=api_key)

# The 3 files Durga bhaiyaa gave you
files_to_process = [
    ("articles_14_07_2026.csv", "Report_14_July.md", "14-07-2026"),
    ("articles_15_07_2026.csv", "Report_15_July.md", "15-07-2026"),
    ("articles_16_07_2026.csv", "Report_16_July.md", "16-07-2026")
]

# Keywords to dynamically filter the noise
bullish_keywords = ["profit", "surge", "jump", "beat", "deal", "revenue", "growth", "approved", "invest", "rally"]
bearish_keywords = ["fall", "crash", "tumble", "slip", "slump", "tension", "crude", "oil", "inflation", "rupee", "war", "blockade", "strike"]

for csv_file, output_md, date_str in files_to_process:
    if not os.path.exists(csv_file):
        print(f"[SKIP] {csv_file} not found.")
        continue

    print(f"\n==========================================")
    print(f"PROCESSING DATASET: {csv_file} ({date_str})")
    print(f"==========================================")

    df = pd.read_csv(csv_file)
    bullish_set = set()
    bearish_set = set()

    # PHASE 1: PYTHON "AI BOUNCER" & DEDUPLICATION
    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        desc = str(row.get('description', ''))
        combined = f"{title}: {desc}"

        # Dynamically categorize based on keywords
        if any(word in combined.lower() for word in bullish_keywords):
            bullish_set.add(title) # Adding just the title natively deduplicates the news!
            
        if any(word in combined.lower() for word in bearish_keywords):
            bearish_set.add(title)

    # Compress payload by taking the top unique headlines
    clean_payload = {
        "date": date_str,
        "bullish_drivers": list(bullish_set)[:7], # Keep top 7 unique bullish news
        "bearish_drivers": list(bearish_set)[:7]  # Keep top 7 unique bearish news
    }

    print(f"Data compressed! Extracted {len(clean_payload['bullish_drivers'])} Bullish and {len(clean_payload['bearish_drivers'])} Bearish unique signals.")

    # PHASE 2: LLAMA-3.3-70B ORCHESTRATION
    prompt = f"""
    You are the Chief Investment Officer analyzing Indian equity markets for {date_str}.

    DEDUPLICATED HIGH-SIGNAL METRICS:
    {json.dumps(clean_payload, indent=2)}

    Provide a highly professional Markdown report with:
    1. DIRECTIONAL CALL: Nifty 50 & Bank Nifty (UP / DOWN / NEUTRAL)
    2. CONFIDENCE RATING (0-100%)
    3. RISK WEIGHTING: Balance Corporate/Earnings (Bullish) vs Macro/Geopolitical risks (Bearish).
    4. KEY CITED METRICS: List specific news events driving your decision.
    """

    try:
        print("Routing to Groq Llama-3.3-70B...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        report_text = response.choices[0].message.content
        
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(f"[SUCCESS] Report saved to {output_md}")

    except Exception as e:
        print(f"[ERROR] Failed to process {csv_file}: {e}")

print("\n[ALL DONE] Your pipeline is complete!")