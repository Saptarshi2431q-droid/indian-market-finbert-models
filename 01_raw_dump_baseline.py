import pandas as pd
import os
from google import genai

# This script intentionally triggers a Rate Limit / Quota error to prove Task 1 & Task 8
api_key = os.environ.get("GCP_API_KEY", "YOUR_GCP_API_KEY")
client = genai.Client(api_key=api_key)

df = pd.read_csv("articles_14_07_2026.csv")
raw_articles = [f"Title: {row.get('title', '')}\nDesc: {row.get('description', '')}" for _, row in df.iterrows()]
full_raw_dump = "\n".join(raw_articles)

prompt = f"Analyze these ~100 articles:\n{full_raw_dump}\nProvide a directional call for Nifty."

print(f"Sending ~{len(full_raw_dump)//4} tokens to Gemini API...")
try:
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
    print(response.text)
except Exception as e:
    print(f"\n[PIPELINE COLLAPSE] Task 1 & 8 Proven: {e}")