import pandas as pd
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
files_to_process = ["articles_14_07_2026.csv", "articles_15_07_2026.csv", "articles_16_07_2026.csv"]

print("=== TASK 1 & 8: RAW DUMP BASELINE TEST ===")

for csv_file in files_to_process:
    if not os.path.exists(csv_file):
        continue
    
    print(f"\nAttempting Raw Dump for: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Dump everything into one massive string
    raw_articles = [f"Title: {row.get('title', '')}\nDesc: {row.get('description', '')}" for _, row in df.iterrows()]
    full_raw_dump = "\n".join(raw_articles)
    
    prompt = f"Analyze these ~100 articles:\n{full_raw_dump}\nProvide a directional call for Nifty."
    
    try:
        # Using the 8B (Haiku Proxy) for the raw dump
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        print("[WARNING] The model somehow survived the raw dump. Signal dilution likely occurred.")
    except Exception as e:
        print(f"[EXPECTED FAILURE] Pipeline collapsed under raw data weight: {e}")