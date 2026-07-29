import os
import json
import pandas as pd

# List your CSV files sitting in your folder
csv_files = [
    'articles_14_07_2026.csv',
    'articles_15_07_2026.csv',
    'articles_16_07_2026.csv'
]

combined_articles = []

for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        # Combine title and description for each article
        for _, row in df.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else ""
            desc = str(row['description']) if pd.notna(row['description']) else ""
            full_text = f"{title}. {desc}".strip()
            if len(full_text) > 10:
                combined_articles.append(full_text)
        print(f" Loaded {len(df)} articles from {file}")
    else:
        print(f" Warning: {file} not found in current directory.")

print(f"\n Total Combined Articles: {len(combined_articles)}")

# Save to raw dump JSON file
with open('real_articles_dump.json', 'w') as f:
    json.dump(combined_articles, f, indent=4)

print(" Saved to real_articles_dump.json")