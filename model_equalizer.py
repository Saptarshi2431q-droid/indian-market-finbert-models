import os
import json
import sys
from groq import Groq

GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    print("[SYSTEM HALT] GROQ_API_KEY not found.")
    sys.exit(1)

client = Groq(api_key=GROQ_KEY)

def run_inference(model_name, payload):
    prompt = f"""
You are an elite quantitative analyst. 
Read the following financial drivers and generate a macro directional call (UP, DOWN, or NEUTRAL).
You MUST explicitly cite the specific numerical metrics (e.g., margins, PEG ratios, or deal sizes) that justify your call.

SMART CHUNKED CONTEXT:
{payload}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, 
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[API CRASH] {str(e)}"

def main():
    print("==================================================")
    print("  TASK 9: THE EQUALIZER (SMART CONTEXT SHOWDOWN)")
    print("==================================================")
    
    # Load the highly deduplicated, chunked context from Task 4
    try:
        with open("final_report_context.json", "r", encoding="utf-8") as f:
            smart_payload = json.dumps(json.load(f))
    except FileNotFoundError:
        print("[ERROR] final_report_context.json not found.")
        return

    models = {
        "Small Model (8B)": "llama-3.1-8b-instant",
        "Large Model (70B)": "llama-3.3-70b-versatile"
    }

    print(f"[PHASE] Feeding the 99.9% reduced Smart Context to both models...\n")
    
    for alias, model_id in models.items():
        print(f"--> Bootstrapping {alias}...")
        output = run_inference(model_id, smart_payload)
        
        if "[API CRASH]" in output:
            print(f"    [{alias}] FAILURE: Pipeline crashed.")
        else:
            # Did they both successfully reason through the data without crashing?
            found_n1 = "21.5" in output or "margin" in output.lower()
            
            print(f"    [{alias}] Execution Complete! No API Crash.")
            print(f"    -> Signal Recall (Needle 1): {found_n1}")
            print(f"    -> Directional Call Preview: {output[:100].strip()}...\n")

if __name__ == "__main__":
    main()