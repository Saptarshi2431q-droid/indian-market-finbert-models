import json
import os
import sys
from groq import Groq

# Ensure Windows terminal handles Unicode characters like ₹ properly
sys.stdout.reconfigure(encoding='utf-8')

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Load clean context
with open('real_report_context.json', 'r', encoding='utf-8') as f:
  context = json.load(f)

prompt = f"""
You are an elite Chief Risk Officer and Quantitative Strategist for Indian Markets.
Analyze the following clean market signals extracted from July 14-16, 2026 news articles:

BULLISH SIGNALS:
{json.dumps(context['unique_bullish_signals'], indent=2)}

BEARISH SIGNALS:
{json.dumps(context['unique_bearish_signals'], indent=2)}

Generate a formal Market Directional Report containing:
1. Directional Call: [UP / DOWN / NEUTRAL]
2. Confidence Score: [0-100%]
3. Key Market Catalysts (Citing specific news facts)
4. Sector-Specific Risk Assessment (Nifty IT, Nifty Bank, Macro)
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,  # Zero temperature for 100% reproducible outputs
)
report = response.choices[0].message.content

# Fix: Added encoding='utf-8' to prevent Windows cp1252 crashes on Rupee (₹) symbols
with open('REAL_NIFTY_MARKET_CALL.md', 'w', encoding='utf-8') as f:
  f.write(report)

print('\n Real Nifty Directional Report Generated Successfully!')
print(' Saved output to REAL_NIFTY_MARKET_CALL.md\n')
print(report)