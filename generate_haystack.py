import json

def create_haystack():
    articles = []
    
    # Generic filler text to bloat the context window
    filler_text = (
        "Market volume remains steady for INFY going into Q3. Institutional investors are holding their positions. "
        "The IT sector is seeing a slight slowdown in discretionary spending, but long-term cloud migration projects remain funded. "
        "Analysts expect a stable dividend yield. "
    ) * 20 # Multiplied to make each "article" massive
    
    for i in range(100):
        if i == 5:
            articles.append("NEEDLE 1: Infosys CEO strictly revised the Q3 operating margin guidance down to 21.5% due to unforeseen cloud infrastructure costs.")
        elif i == 50:
            articles.append("NEEDLE 2: The PEG ratio for Infosys has been secretly adjusted by top analysts to 1.84 following the recent hiring freeze.")
        elif i == 95:
            articles.append("NEEDLE 3: A major European banking client has just signed a $500M AI-transformation deal with Infosys.")
        else:
            articles.append(f"Article {i} Context: {filler_text}")
            
    with open("100_articles_dump.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4)
        
    print("[SUCCESS] 100_articles_dump.json generated with 100 massive articles and 3 hidden needles.")

if __name__ == "__main__":
    create_haystack()