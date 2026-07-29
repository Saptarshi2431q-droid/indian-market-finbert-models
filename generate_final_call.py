import os
import json
import sys
from groq import Groq

GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    print("[ERROR] GROQ_API_KEY not found in environment variables.")
    sys.exit(1)

client = Groq(api_key=GROQ_KEY)

def run_final_quantum_call():
    print("[INIT] Ingesting deduplicated financial context...")
    try:
        with open("final_report_context.json", "r", encoding="utf-8") as f:
            context_data = json.load(f)
    except FileNotFoundError:
        print("[ERROR] final_report_context.json not found. Run assemble_context.py first.")
        return

    # Programmatic Context Injection Strategy (Task 4/5 integration)
    bullish_str = "\n".join([f"- {s}" for s in context_data["BULLISH_DRIVERS"]])
    bearish_str = "\n".join([f"- {s}" for s in context_data["BEARISH_DRIVERS"]])

    system_prompt = (
        "You are the Lead Quantitative Risk Director at AI4Invest.\n"
        "Your task is to analyze the provided Bullish and Bearish drivers, synthesize them, "
        "and issue a definitive directional macro call for Nifty and Bank Nifty.\n\n"
        "You must output your analysis in strict institutional Markdown format matching the company standard."
    )

    user_payload = f"""
Analyze these isolated high-signal market drivers:

### EXTRACTED BULLISH DRIVERS:
{bullish_str if bullish_str else "- No high-confidence bullish drivers detected."}

### EXTRACTED BEARISH DRIVERS:
{bearish_str if bearish_str else "- No high-confidence bearish drivers detected."}

Provide a definitive trading thesis. You must explicitly state:
1. Directional Macro Call (UP / DOWN / NEUTRAL)
2. Quantitative Confidence Level (0% to 100%) based on the signal weight
3. Concrete Citations of the specific drivers that forced your decision.
"""

    print("[PROCESS] Dispatching high-signal tokens to Llama-3.3-70B on Groq LPU...")
    try:
        api_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.2, # Low temperature to enforce strict logical synthesis
            max_tokens=4000
        )
        
        trading_thesis = api_response.choices[0].message.content
        
        # Save the finalized institutional report
        output_filename = "NIFTY_Final_Directional_Call.md"
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(trading_thesis)
            
        print("\n" + "="*50)
        print(f"[SUCCESS] Macro Directional Thesis Generated Successfully!")
        print(f"Saved locally as: {output_filename}")
        print("="*50)

    except Exception as e:
        print(f"[ERROR] Groq API execution failed: {str(e)}")

if __name__ == "__main__":
    run_final_quantum_call()