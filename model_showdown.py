import os
import json
import sys
from groq import Groq

# Ensure the Groq Key is loaded
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    print("\n[SYSTEM HALT] GROQ_API_KEY not found in environment variables.")
    print("Run: $env:GROQ_API_KEY=\"your_key_here\" in PowerShell.")
    sys.exit(1)

client = Groq(api_key=GROQ_KEY)

def run_inference(model_name, payload, context_size_name):
    prompt = f"""
You are an elite quantitative analyst. 
Read the following raw financial text and generate a macro directional call (UP, DOWN, or NEUTRAL).
You MUST explicitly cite the specific numerical metrics (e.g., margins, PEG ratios, or deal sizes) that justify your call.

RAW TEXT:
{payload}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Zero entropy to strictly test logical recall
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[API CRASH] {str(e)}"

def main():
    print("==================================================")
    print("  IGNITING WORKSTREAM B: SLM VS LLM SCALING AUDIT")
    print("==================================================")
    
    # 1. Load the RAW, UNCHUNKED haystack (we are testing raw context degradation)
    try:
        with open("100_articles_dump.json", "r", encoding="utf-8") as f:
            raw_articles = json.load(f)
    except FileNotFoundError:
        print("[ERROR] 100_articles_dump.json not found.")
        return

    # 2. Build Context Payloads
    # SMALL CONTEXT: First 10 articles. Only contains Needle 1 (21.5% margin drop).
    small_context = json.dumps(raw_articles[:10]) 
    
    # LARGE CONTEXT: All 100 articles. Contains Needle 1, Needle 2 (1.84 PEG), and Needle 3 ($500M AI Deal).
    large_context = json.dumps(raw_articles) 

    # 3. Model Matrix mapped to Groq
    models = {
        "Small Model (8B)": "llama-3.1-8b-instant",
        "Large Model (70B)": "llama-3.3-70b-versatile"
    }

    # 4. Execute the Showdown
    results = {}
    for context_name, payload in [("SMALL (10 Articles)", small_context), ("LARGE (100 Articles)", large_context)]:
        print(f"\n[PHASE] Deploying {context_name} Payload...")
        results[context_name] = {}
        
        for alias, model_id in models.items():
            print(f"--> Bootstrapping {alias}...")
            output = run_inference(model_id, payload, context_name)
            results[context_name][alias] = output
            
            # MLOps Telemetry: Did they actually recall the specific financial metrics?
            # Or did the API crash entirely?
            if "[API CRASH]" in output:
                print(f"    [{alias}] FAILURE: Pipeline collapsed under payload weight.")
            else:
                found_n1 = "21.5" in output or "margin" in output.lower()
                found_n2 = "1.84" in output or "peg" in output.lower()
                found_n3 = "500" in output or "european" in output.lower()
                
                print(f"    [{alias}] Execution Complete.")
                print(f"    -> Signal Recall: Needle 1 (Margin): {found_n1} | Needle 2 (PEG): {found_n2} | Needle 3 (AI Deal): {found_n3}")

    # Save detailed logs for the final report
    with open("model_showdown_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\n[SUCCESS] Matrix execution complete. Output saved to model_showdown_results.json")

if __name__ == "__main__":
    main()