"""
Phase 2: Western FinBERT 100-Headline Spatial Audit
Author: Rishi | AI4Invest Macro Supervisor Pipeline
Objective: Evaluate vector space morphing and OOD semantic drift using ProsusAI/finbert.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

def generate_100_headlines_dataset():
    """Programmatically generates the exact same 100-headline balanced dataset."""
    
    macro_bullish = [
        "RBI cuts repo rate by 25 bps, injecting massive liquidity.", "Inflation drops below target, signaling rate easing.",
        "GDP growth beats estimates, accelerating to 7.2 percent.", "Monetary Policy Committee signals dovish stance.",
        "Forex reserves hit all-time high, strengthening Rupee.", "Government announces massive infrastructure spending package.",
        "Tax collection run rate outpaces budgetary estimates.", "Central bank intervenes to stabilize bond yields successfully.",
        "Export volumes surge despite global macroeconomic headwinds.", "Fiscal deficit narrows significantly in Q3.",
        "Industrial production hits record high, beating all estimates.", "RBI governor guarantees systemic liquidity support.",
        "Monsoon surplus drives strong rural consumption recovery."
    ] 
    macro_bearish = [
        "RBI unexpectedly hikes CRR by 50 bps to drain liquidity.", "Inflation prints higher than expected, forcing RBI intervention.",
        "Monetary Policy Committee maintains withdrawal of accommodation.", "Rupee hits record low against the US Dollar.",
        "Bond yields spike dangerously on sovereign debt concerns.", "GDP growth contracts sharply amid global recession fears.",
        "Current account deficit widens to alarming levels.", "Central bank warns of persistent systemic credit risks.",
        "Trade deficit expands as imports heavily outpace exports.", "Unemployment data indicates severe macroeconomic slowdown.",
        "Rating agencies downgrade sovereign outlook to negative.", "Liquidity deficit in banking system widens to record highs."
    ] 

    earnings_bullish = [
        "Reliance posts record quarterly profit driven by Jio.", "TCS beats street estimates with massive margin expansion.",
        "HDFC Bank reports robust credit growth and lower NPAs.", "Infosys raises annual revenue guidance on strong deal wins.",
        "ICICI Bank net interest margins hit multi-year highs.", "Tata Motors reports explosive growth in EV sales volume.",
        "L&T order book crosses historic milestone in Q4.", "SBI posts highest ever quarterly net profit.",
        "Maruti Suzuki margins expand on lower commodity costs.", "Wipro announces massive share buyback following strong results.",
        "ITC declares special dividend after beating street estimates.", "Sun Pharma US sales surge, driving record profitability.",
        "Bajaj Finance AUM growth accelerates past expectations."
    ] 
    earnings_bearish = [
        "Infosys misses Q3 revenue guidance, shares plummet.", "HDFC Bank reports rising NPAs, forcing increased provisions.",
        "TCS margins contract heavily on rising employee costs.", "Reliance retail growth slows down significantly.",
        "Tech Mahindra posts catastrophic drop in quarterly profit.", "Tata Steel reports massive loss on European operations.",
        "SBI asset quality deteriorates, triggering massive sell-off.", "Wipro cuts revenue guidance amid macroeconomic uncertainty.",
        "Maruti Suzuki volumes drop due to semiconductor shortages.", "L&T infrastructure margins compress under inflation pressure.",
        "ITC cigarette volumes decline sharply on new tax hikes.", "Bajaj Finance reports severe slowdown in new customer acquisitions."
    ] 

    ma_bullish = [
        "Tata Sons completes massive acquisition of European tech firm.", "HDFC twin merger finalized, creating global banking giant.",
        "Reliance acquires major green energy startup to expand portfolio.", "Adani successfully completes buyout of major cement assets.",
        "Axis Bank acquires Citi's consumer business smoothly.", "Infosys acquires prominent AI firm to boost cloud capabilities.",
        "Zomato acquires Blinkit, expanding rapid commerce dominance.", "M&M signs joint venture with global EV manufacturer.",
        "TCS acquires advanced European design firm.", "Sun Pharma completes strategic acquisition in dermatology.",
        "Biocon acquires Viatris biosimilars business globally.", "PVR and INOX merger cleared by competition commission.",
        "LTI and Mindtree successfully complete integration process."
    ] 
    ma_bearish = [
        "Regulator heavily blocks proposed merger between Zee and Sony.", "Hostile takeover bid fails completely due to lack of support.",
        "Tata Motors cancels planned subsidiary spin-off.", "Competition Commission imposes massive penalty, halting acquisition.",
        "Reliance walks away from highly anticipated retail acquisition.", "HDFC merger faces severe regulatory delays.",
        "Adani acquisition probed by market regulator.", "Axis Bank integration costs spiral out of control.",
        "Zomato acquisition highly criticized by institutional investors.", "M&M joint venture collapses over valuation disputes.",
        "TCS acquisition target faces massive internal fraud probe.", "Biocon acquisition runs into heavy FDA regulatory hurdles."
    ] 

    sector_bullish = [
        "IT sector witnesses massive influx of foreign institutional buying.", "Banking index hits all-time high on robust credit cycle.",
        "FMCG sector volumes show early signs of stabilizing.", "Automobile sector reports explosive festive demand pickup.",
        "Pharma stocks rally on strong US FDA approvals.", "Metal index surges on rebounding global commodity prices.",
        "Real estate sector records highest quarterly sales in a decade.", "Infrastructure sector benefits from massive budget allocations.",
        "Energy stocks rally on stable crude oil prices.", "Midcap stocks witness intense buying from retail investors.",
        "Defense sector order books swell on government push.", "Capital goods sector sets up structural multi-year rally.",
        "PSU Bank index breaks past historic resistance levels."
    ] 
    sector_bearish = [
        "IT sector faces intense selling pressure on US recession fears.", "Banking index crashes as leverage unwinding triggers margin calls.",
        "FMCG sector margins compress heavily under rural slowdown.", "Automobile sector faces severe semiconductor supply chain shocks.",
        "Pharma stocks plummet following multiple US FDA warning letters.", "Metal index crashes on fears of global demand drop.",
        "Real estate sector stalls as interest rates remain elevated.", "Infrastructure projects face massive execution delays.",
        "Energy sector margins wiped out by volatile crude prices.", "Smallcap stocks completely dump as retail public runs away.",
        "Defense sector stocks crash on delayed government payments.", "PSU Bank index heavily shorted by foreign institutional investors."
    ] 

    data = []
    def build_rows(headline_list, topic, sentiment):
        for text in headline_list:
            data.append({"Text": text, "Topic": topic, "Sentiment": sentiment})

    build_rows(macro_bullish, "Macro Policy", "Bullish")
    build_rows(macro_bearish, "Macro Policy", "Bearish")
    build_rows(earnings_bullish, "Earnings", "Bullish")
    build_rows(earnings_bearish, "Earnings", "Bearish")
    build_rows(ma_bullish, "M&A", "Bullish")
    build_rows(ma_bearish, "M&A", "Bearish")
    build_rows(sector_bullish, "Sector Trends", "Bullish")
    build_rows(sector_bearish, "Sector Trends", "Bearish")

    return pd.DataFrame(data)

def main():
    print("Loading Western FinBERT (ProsusAI/finbert)...")
    # We load the base model architecture of FinBERT to extract the raw embeddings, 
    # proving the vector space itself has been warped by the training data.
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()

    df = generate_100_headlines_dataset()
    print(f"Dataset Loaded: {len(df)} Financial Headlines.")

    embeddings = []
    print("Extracting 768-D [CLS] vectors from FinBERT...")
    
    with torch.no_grad():
        for text in df['Text']:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
            
    embeddings = np.array(embeddings)

    print("\nApplying PCA Dimensionality Reduction...")
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)

    df['PCA_X'] = reduced_embeddings[:, 0]
    df['PCA_Y'] = reduced_embeddings[:, 1]
    
    # --- SILHOUETTE SCORES ---
    topic_labels = pd.factorize(df['Topic'])[0]
    sentiment_labels = pd.factorize(df['Sentiment'])[0]
    
    topic_score = silhouette_score(embeddings, topic_labels)
    sentiment_score = silhouette_score(embeddings, sentiment_labels)
    
    print("\n" + "="*50)
    print("   FINBERT VECTOR SEPARATION METRICS (100 SAMPLES)")
    print("="*50)
    print(f"Topic Separation Score     : {topic_score:.4f} (Notice the collapse)")
    print(f"Sentiment Separation Score : {sentiment_score:.4f} (Notice the surge)")
    print("="*50)

    # --- Plotting ---
    print("\nGenerating Scatter Plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Phase 2: Western FinBERT Processing 100 Indian Headlines', fontsize=18, fontweight='bold')

    # Plot 1: Sentiment
    sns.scatterplot(data=df, x='PCA_X', y='PCA_Y', hue='Sentiment', 
                    palette={'Bullish': '#2ca02c', 'Bearish': '#d62728'}, 
                    s=100, alpha=0.8, ax=axes[0], edgecolor='black')
    axes[0].set_title(f'Colored by True Sentiment\nSilhouette Score: {sentiment_score:.4f}', fontsize=14)
    axes[0].set_xlabel('PCA Dimension 1')
    axes[0].set_ylabel('PCA Dimension 2')
    
    # Plot 2: Topic
    sns.scatterplot(data=df, x='PCA_X', y='PCA_Y', hue='Topic', 
                    palette='Dark2', s=100, alpha=0.8, ax=axes[1], edgecolor='black')
    axes[1].set_title(f'Colored by Topic\nSilhouette Score: {topic_score:.4f}', fontsize=14)
    axes[1].set_xlabel('PCA Dimension 1')
    axes[1].set_ylabel('PCA Dimension 2')

    plt.tight_layout()
    plt.savefig("finbert_100_analysis.png", dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved 'finbert_100_analysis.png' to workspace.")
    plt.show()

if __name__ == "__main__":
    main()