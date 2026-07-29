import json

with open('finbert_raw_sentences.json', 'r', encoding='utf-8') as f:
  data = json.load(f)

# Sort unique sentences deterministically so results never change randomly
unique_bullish = sorted(list(set(data['bullish'])))
unique_bearish = sorted(list(set(data['bearish'])))

# Select the top 15 highest-signal sentences consistently
clean_payload = {
    'unique_bullish_signals': unique_bullish[:15],
    'unique_bearish_signals': unique_bearish[:15],
}

with open('real_report_context.json', 'w', encoding='utf-8') as f:
  json.dump(clean_payload, f, indent=4)

print(" Saved deterministic context to real_report_context.json")