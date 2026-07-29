import json
import os
import sys
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Load vector-retrieved banking documents
with open('nifty_bank_vector_context.json', 'r', encoding='utf-8') as f:
  context = json.load(f)

prompt = f"""
You are an Senior Banking Sector Analyst specializing in NIFTY BANK and Indian Financial Institutions.
Analyze these vector-retrieved market documents from July 14-16, 2026:

RETRIEVED BANKING DOCUMENTS FROM VECTOR DB:
{json.dumps(context['retrieved_chunks'], indent=2)}

Generate a dedicated **NIFTY BANK SECTOR REPORT** containing:
1. Sector Stance: [BULLISH / BEARISH / NEUTRAL]
2. Key Banking Catalysts & Earnings Drivers (HDFC, ICICI, Axis, SBI, etc.)
3. Systemic Risks (Credit growth, RBI policy, non-performing assets, litigation)
4. Tactical Trading Outlook for NIFTY BANK Index
"""

response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0,
)

report = response.choices[0].message.content

with open('NIFTY_BANK_VECTOR_REPORT.md', 'w', encoding='utf-8') as f:
  f.write(report)

print('\n Nifty Bank Vector RAG Report Generated Successfully!')
print(' Saved output to NIFTY_BANK_VECTOR_REPORT.md\n')
print(report)