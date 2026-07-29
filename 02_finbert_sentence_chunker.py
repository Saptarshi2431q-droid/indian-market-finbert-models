import json
import re
import torch
from transformers import pipeline

# Automatically detect GPU vs CPU
device_id = 0 if torch.cuda.is_available() else -1

print(" Loading FinBERT model...")
classifier = pipeline(
    "text-classification",
    model="sapt3009/AI4Invest-Indian-FinBERT",
    device=device_id,
)

# Load raw dump
with open("real_articles_dump.json", "r", encoding="utf-8") as f:
  articles = json.load(f)

# Step 1: Collect and truncate all sentences first
all_sentences = []
print(" Splitting 1,176 articles into sentences...")
for text in articles:
  sentences = re.split(r"(?<=[.!?]) +", text)
  for sentence in sentences:
    sentence = sentence.strip()
    if len(sentence.split()) > 5:
      all_sentences.append(sentence[:512])

total_sentences = len(all_sentences)
print(f" Extracted {total_sentences} sentences to classify.")

# Step 2: Run inference in batches of 32 (10x-20x faster)
batch_size = 32
bullish_sentences = []
bearish_sentences = []

print(" Running FinBERT inference in batches...\n")

for i in range(0, total_sentences, batch_size):
  batch = all_sentences[i : i + batch_size]
  results = classifier(batch)

  for sentence, res in zip(batch, results):
    label = res["label"].lower()
    score = res["score"]

    if score > 0.85:
      if label == "positive":
        bullish_sentences.append(sentence)
      elif label == "negative":
        bearish_sentences.append(sentence)

  # Live progress indicator
  processed = min(i + batch_size, total_sentences)
  pct = int((processed / total_sentences) * 100)
  print(
      f" Progress: {processed}/{total_sentences} sentences ({pct}%)", end="\r"
  )

print(
    f"\n\n Finished! Extracted {len(bullish_sentences)} Bullish and"
    f" {len(bearish_sentences)} Bearish sentences."
)

# Save raw sentence predictions
with open("finbert_raw_sentences.json", "w", encoding="utf-8") as f:
  json.dump(
      {"bullish": bullish_sentences, "bearish": bearish_sentences},
      f,
      indent=4,
  )

print(" Saved results to finbert_raw_sentences.json")