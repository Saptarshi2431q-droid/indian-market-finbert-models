"""
Project: AI4Invest MLOps Agent Framework
Module: Core Cognitive Orchestration Engine
Author: Saptarshi Dutta (Rishi)
"""

import os
import json
import sys
from groq import Groq
import agent_tools 

GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    print("[ERROR] GROQ_API_KEY not found.")
    sys.exit(1)

client = Groq(api_key=GROQ_KEY)

agent_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_live_price",
            "description": "Fetch the live stock price for an Indian NSE ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker, e.g., INFY"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_news",
            "description": "Retrieve the latest financial news headlines for a stock ticker. Returns a list of text strings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker, e.g., INFY"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stock_sentiment",
            "description": "Run custom FinBERT sentiment analysis on a list of news headlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "headlines_json": {
                        "type": "string", 
                        "description": "The JSON array of headline strings returned by the get_financial_news tool."
                    }
                },
                "required": ["headlines_json"]
            }
        }
    }
]

def execute_quant_agent_loop(user_query: str):
    print("\n" + "=" * 75)
    print(f"BOOTSTRAPPING QUANT COGNITIVE RUN: {user_query}")
    print("=" * 75)
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an elite Institutional Quantitative Risk Agent. "
                "Step 1: Fetch the live price and the news. "
                "Step 2: Take the exact output from the news tool and pass it into analyze_stock_sentiment. "
                "Step 3: Synthesize all data into a highly professional markdown report."
            )
        },
        {"role": "user", "content": user_query}
    ]
    
    for current_step in range(5):
        print(f"\n[ORCHESTRATION LOOP STEP {current_step + 1}] Processing token embeddings and planning action...")
        
        api_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=messages,
            tools=agent_schema,
            tool_choice="auto",
            parallel_tool_calls=False # Prevents syntax hallucination
        )
        
        llm_decision = api_response.choices[0].message
        messages.append(llm_decision) 
        
        if not llm_decision.tool_calls:
            print("[ORCHESTRATION COMPLETE] Cognitive pipeline has synthesized all runtime observations.")
            return llm_decision.content
            
        for targeted_call in llm_decision.tool_calls:
            execution_name = targeted_call.function.name
            execution_args = json.loads(targeted_call.function.arguments)
            print(f"--> Match Verified. Triggering Python Interface: {execution_name}")
            
            if execution_name == "get_live_price":
                observation = agent_tools.get_live_price(execution_args.get("ticker"))
            elif execution_name == "get_financial_news":
                observation = agent_tools.get_financial_news(execution_args.get("ticker"))
            elif execution_name == "analyze_stock_sentiment":
                observation = agent_tools.analyze_stock_sentiment(execution_args.get("headlines_json"))
            else:
                observation = json.dumps({"error": "Unknown tool."})
                
            print(f"<-- Metric Captured: {str(observation)[:80]}...")
            
            messages.append({
                "role": "tool",
                "tool_call_id": targeted_call.id,
                "name": execution_name,
                "content": observation
            })
            
    return "[SYSTEM HALT] Execution limit reached."

if __name__ == "__main__":
    target_prompt = "Generate a quantitative risk report on INFY (Infosys). Fetch its price, news, and run the FinBERT sentiment analysis."
    trading_thesis = execute_quant_agent_loop(target_prompt)
    
    report_filename = "INFY_Quantitative_Risk_Report.md"
    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(f"# AI4Invest Autonomous Execution Engine\n\n")
        file.write(trading_thesis)
        
    print("\n" + "=" * 75)
    print(f"[SUCCESS] Proprietary Quant Trading Thesis generated.")
    print(f"Report saved locally as: {report_filename}")
    print("Open this file in VS Code and press (Ctrl + Shift + V) to view the formatted report.")
    print("=" * 75)