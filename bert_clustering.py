"""
Base BERT Clustering Analysis - AI4Invest
Author: Rishi | Macro Supervisor Pipeline
Objective: Prove that bert-base-uncased clusters by topic, not sentiment.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

def main():
    print("Loading bert-base-uncased tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    # Sample dataset engineered to test Topic vs. Sentiment separation
    data = [
        # Topic: RBI / Macro
        {"text": "RBI unexpectedly hikes CRR by 50 bps to drain banking liquidity.", "Sentiment": "Bearish", "Topic": "Macro Policy"},
        {"text": "Monetary Policy Committee holds repo rate steady, signals robust growth.", "Sentiment": "Bullish", "Topic": "Macro Policy"},
        {"text": "Inflation data prints higher than expected, RBI intervention likely.", "Sentiment": "Bearish", "Topic": "Macro Policy"},
        {"text": "Governor Das highlights strong systemic liquidity in latest address.", "Sentiment": "Bullish", "Topic": "Macro Policy"},
        
        # Topic: Corporate Earnings
        {"text": "Reliance posts record quarterly profit driven by strong Jio subscriber additions.", "Sentiment": "Bullish", "Topic": "Earnings"},
        {"text": "Infosys misses Q3 revenue guidance, shares plummet on Dalal Street.", "Sentiment": "Bearish", "Topic": "Earnings"},
        {"text": "TCS beats street estimates with massive margin expansion.", "Sentiment": "Bullish", "Topic": "Earnings"},
        {"text": "HDFC Bank reports rising NPAs, forces increased provisions.", "Sentiment": "Bearish", "Topic": "Earnings"},
        
        # Topic: Mergers & Acquisitions
        {"text": "Tata Motors acquires massive European manufacturing plant to expand EV footprint.", "Sentiment": "Bullish", "Topic": "M&A"},
        {"text": "Regulator heavily blocks the proposed merger between Zee and Sony.", "Sentiment": "Bearish", "Topic": "M&A"},
        {"text": "Adani successfully completes acquisition of major cement assets.", "Sentiment": "Bullish", "Topic": "M&A"},
        {"text": "Hostile takeover bid fails completely due to lack of shareholder support.", "Sentiment": "Bearish", "Topic": "M&A"},
    ]

    df = pd.DataFrame(data)
    embeddings = []

    print("\nExtracting [CLS] token embeddings from financial headlines...")
    with torch.no_grad():
        for text in df['text']:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            
            # Extract the [CLS] token (Index 0 of the last hidden state)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding.flatten())

    print("Applying PCA to reduce 768 dimensions to 2D for visualization...")
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)

    df['PCA_X'] = reduced_embeddings[:, 0]
    df['PCA_Y'] = reduced_embeddings[:, 1]
    
    print("\nGenerating Scatter Plots...")
    
    # Configure Matplotlib Figure
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Base BERT [CLS] Embeddings: Topic vs Sentiment', fontsize=18, fontweight='bold')

    # Plot 1: Colored by Sentiment (Should look mixed/random)
    sns.scatterplot(data=df, x='PCA_X', y='PCA_Y', hue='Sentiment', 
                    palette={'Bullish': '#2ca02c', 'Bearish': '#d62728'}, 
                    s=150, ax=axes[0], edgecolor='black')
    axes[0].set_title('Colored by Sentiment\n(Notice the lack of separation)', fontsize=14)
    axes[0].set_xlabel('PCA Dimension 1')
    axes[0].set_ylabel('PCA Dimension 2')
    
    # Plot 2: Colored by Topic (Should look cleanly clustered)
    sns.scatterplot(data=df, x='PCA_X', y='PCA_Y', hue='Topic', 
                    palette='Dark2', s=150, ax=axes[1], edgecolor='black')
    axes[1].set_title('Colored by Topic\n(Notice the clear spatial grouping)', fontsize=14)
    axes[1].set_xlabel('PCA Dimension 1')
    axes[1].set_ylabel('PCA Dimension 2')

    plt.tight_layout()
    
    # Save the output for Durga Bhaiyaa's presentation
    output_filename = "bert_base_clusters.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Visual artifact saved to workspace as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()