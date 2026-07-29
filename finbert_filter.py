import json
import re
from transformers import pipeline

def build_chunked_smart_context():
    print("[INIT] Loading proprietary AI4Invest-Indian-FinBERT pipeline...")
    try:
        sentiment_pipeline = pipeline("text-classification", model="sapt3009/AI4Invest-Indian-FinBERT")
    except Exception as e:
        print(f"[ERROR] Failed to load FinBERT: {e}")
        return

    print("[PROCESS] Reading the 100-article haystack...")
    try:
        with open("100_articles_dump.json", "r", encoding="utf-8") as f:
            raw_articles = json.load(f)
    except FileNotFoundError:
        print("[ERROR] 100_articles_dump.json not found. Run generate_haystack.py first.")
        return

    filtered_chunks = []
    print("[PROCESS] Chunking articles at the sentence level and scoring each chunk...")
    
    for article_idx, text in enumerate(raw_articles):
        # ---------------------------------------------------------
        # TASK 3 CHUNKING ARCHITECTURE:
        # Splitting the massive article into individual sentences
        # ---------------------------------------------------------
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        for chunk_idx, sentence in enumerate(sentences):
            # Clean up whitespace and drop tiny, meaningless fragments
            clean_sentence = sentence.strip()
            if len(clean_sentence) < 15: 
                continue
            
            # Score the isolated chunk
            result = sentiment_pipeline(clean_sentence[:512])[0]
            label = result['label'].lower()
            score = result['score']

            # Diagnostic check: Did we successfully isolate our Needles?
            if "NEEDLE" in clean_sentence:
                print(f"--> NEEDLE ISOLATED (Article {article_idx}): Classified as {label} with {score:.4f} confidence")

            # We can use a stricter threshold (0.60) now that the signal is pure
            if label in ["positive", "negative"] and score > 0.60:
                sentiment_tag = "Bullish" if label == "positive" else "Bearish"
                
                filtered_chunks.append({
                    "article_index": article_idx,
                    "chunk_index": chunk_idx,
                    "sentiment": sentiment_tag,
                    "confidence": round(score, 4),
                    "text": clean_sentence
                })

    print(f"\n[METRICS] Original Haystack: {len(raw_articles)} articles.")
    print(f"[METRICS] Filtered High-Signal Chunks: {len(filtered_chunks)} sentences.")
    
    # Save the highly concentrated signal payload
    with open("chunked_context_payload.json", "w", encoding="utf-8") as f:
        json.dump(filtered_chunks, f, indent=4)
        
    print("[SUCCESS] chunked_context_payload.json generated. Context is ready for Groq.")

if __name__ == "__main__":
    build_chunked_smart_context()