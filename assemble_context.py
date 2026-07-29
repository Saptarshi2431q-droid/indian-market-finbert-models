import json

def assemble_and_deduplicate():
    print("[INIT] Loading chunked context payload...")
    try:
        with open("chunked_context_payload.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("[ERROR] chunked_context_payload.json not found.")
        return

    print(f"[PROCESS] Analyzing {len(chunks)} raw chunks for deduplication...")

    # We use Python 'sets' because they mathematically cannot contain duplicate values
    unique_bullish = set()
    unique_bearish = set()

    for chunk in chunks:
        text = chunk["text"]
        sentiment = chunk["sentiment"]

        if sentiment == "Bullish":
            unique_bullish.add(text)
        elif sentiment == "Bearish":
            unique_bearish.add(text)

    print("\n" + "="*50)
    print("[METRICS] Context Assembly Complete")
    print("="*50)
    print(f"Original Overloaded Payload  : {len(chunks)} sentences")
    print(f"Unique Bullish Signals     : {len(unique_bullish)} sentences")
    print(f"Unique Bearish Signals     : {len(unique_bearish)} sentences")
    
    total_unique = len(unique_bullish) + len(unique_bearish)
    print(f"Context Reduction Achieved : {(1 - (total_unique / len(chunks))) * 100:.2f}% redundant data purged")

    # Rollup by sentiment bucket
    final_context = {
        "BULLISH_DRIVERS": list(unique_bullish),
        "BEARISH_DRIVERS": list(unique_bearish)
    }

    with open("final_report_context.json", "w", encoding="utf-8") as f:
        json.dump(final_context, f, indent=4)

    print("\n[SUCCESS] final_report_context.json generated.")
    print("[SUCCESS] The payload is now deduplicated, bucketed, and Groq-ready!")

if __name__ == "__main__":
    assemble_and_deduplicate()