import pandas as pd
import json
import os
from groq import Groq

# 1. Initialize Groq Client
api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
client = Groq(api_key=api_key)

files_to_process = [
    ("articles_14_07_2026.csv", "Report_2026_07_14.md", "14-07-2026"),
    ("articles_15_07_2026.csv", "Report_2026_07_15.md", "15-07-2026"),
    ("articles_16_07_2026.csv", "Report_2026_07_16.md", "16-07-2026")
]

for csv_file, output_md, date_str in files_to_process:
    if not os.path.exists(csv_file):
        print(f"[SKIP] {csv_file} not found in directory.")
        continue

    print(f"\n==========================================")
    print(f"PROCESSING DATASET: {csv_file} ({date_str})")
    print(f"==========================================")

    df = pd.read_csv(csv_file)
    bullish_set = set()
    bearish_set = set()

    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        desc = str(row.get('description', ''))
        combined = f"{title}: {desc}"

        # High-signal key metric extraction logic
        if any(w in combined for m in ["HCL", "TCS", "Infosys", "Wipro", "profit", "beat", "deal"] for w in [m]):
            bullish_set.add(f"Corporate/IT: {title} - {desc[:120]}...")
        if any(w in combined for m in ["inflation", "rupee", "crude", "oil", "Hormuz", "war", "tariff"] for w in [m]):
            bearish_set.add(f"Macro/Risk: {title} - {desc[:120]}...")

    # Keep top high-signal metrics
    clean_payload = {
        "date": date_str,
        "bullish_signals": list(bullish_set)[:5],
        "bearish_signals": list(bearish_set)[:5]
    }

    prompt = f"""
    You are the Chief Investment Officer analyzing Indian equity markets for {date_str}.

    DEDUPLICATED HIGH-SIGNAL METRICS:
    {json.dumps(clean_payload, indent=2)}

    Provide:
    1. DIRECTIONAL CALL: Nifty 50 & Bank Nifty (UP / DOWN / NEUTRAL)
    2. CONFIDENCE RATING (0-100%)
    3. RISK WEIGHTING: Balance Corporate/Earnings vs Macro/Geopolitical risks.
    4. KEY CITED METRICS: List specific numbers.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        report_text = response.choices[0].message.content
        
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(f"[SUCCESS] Report generated and saved to {output_md}")

    except Exception as e:
        print(f"[ERROR] Failed to process {csv_file}: {e}")

print("\n[ALL DONE] All datasets processed successfully!")