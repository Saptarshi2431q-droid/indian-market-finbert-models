"""
FinBERT Indian Market Empirical Risk Audit
Author: Technical Internship Portfolio
Objective: Run FinBERT on 50 localized headlines and calculate systematic error rates.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def load_audit_dataset():
    # 50 Curated Indian Financial Headlines across 4 distinct local blind spots
    # Categories: 1 = RBI Policy, 2 = Corporate/NSE Disclosures, 3 = Media Idioms, 4 = Hinglish Retail
    return [
        # --- BLIND SPOT 1: RBI POLICY & MONETARY FRAMEWORKS ---
        {"text": "RBI mandates surprise 50 bps CRR hike to drain banking system liquidity.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "Monetary Policy Committee votes unanimously to maintain withdrawal of accommodation stance.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "RBI hikes repo rate by 25 basis points; home loans expected to get costlier.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "State Bank of India raises its MCLR by 10 bps across all tenors immediately.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "RBI holds reverse repo rate steady at 3.35% to anchor short term yields.", "true_sentiment": "neutral", "category": "RBI Policy"},
        {"text": "Commercial banks scramble as RBI tightens risk weights for unsecured consumer retail loans.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "Governor signals readiness to deploy open market operations if system liquidity dries up.", "true_sentiment": "neutral", "category": "RBI Policy"},
        {"text": "RBI relaxes SLR maintenance rules, providing breathing room for mid-tier banks.", "true_sentiment": "positive", "category": "RBI Policy"},
        {"text": "Liquidity deficit in the banking system widens to a record 2.5 lakh crore.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "Inward remittances hit record highs, easing balance of payment pressures.", "true_sentiment": "positive", "category": "RBI Policy"},
        {"text": "RBI Deputy Governor expresses deep concern over inflating systemic credit flags.", "true_sentiment": "negative", "category": "RBI Policy"},
        {"text": "Clean note policy framework extended; banks ordered to upgrade sorting infrastructure.", "true_sentiment": "neutral", "category": "RBI Policy"},

        # --- BLIND SPOT 2: NSE/BSE CORPORATE DISCLOSURES & SEBI TEMPLATES ---
        {"text": "Company clarifies on market rumors regarding massive promoter stake sale.", "true_sentiment": "neutral", "category": "Corporate Disclosures"},
        {"text": "Intimation under Regulation 30 of SEBI LODR: Execution of binding business agreement.", "true_sentiment": "positive", "category": "Corporate Disclosures"},
        {"text": "BSE queries infrastructure conglomerate over sudden and unexplained price volume movement.", "true_sentiment": "negative", "category": "Corporate Disclosures"},
        {"text": "Board approves allotment of equity shares on preferential basis to sovereign wealth funds.", "true_sentiment": "positive", "category": "Corporate Disclosures"},
        {"text": "Disclosure under Regulation 29 of SEBI SAST: Institutional investors dump substantial holdings.", "true_sentiment": "negative", "category": "Corporate Disclosures"},
        {"text": "Company schedules analyst call to explain restructuring plans post NCLT admission.", "true_sentiment": "negative", "category": "Corporate Disclosures"},
        {"text": "Outcome of Board Meeting: Audited financial results for the quarter submitted safely.", "true_sentiment": "neutral", "category": "Corporate Disclosures"},
        {"text": "Promoters complete technical transfer of entire shareholding to holding trust entities.", "true_sentiment": "neutral", "category": "Corporate Disclosures"},
        {"text": "SEBI issues descriptive administrative warning letter to pharma major over clinical disclosure delay.", "true_sentiment": "negative", "category": "Corporate Disclosures"},
        {"text": "Company enters into a comprehensive settlement agreement to clear pending tax litigation.", "true_sentiment": "positive", "category": "Corporate Disclosures"},
        {"text": "Anomalies detected in internal ledger books; statutory auditors issue qualified opinion.", "true_sentiment": "negative", "category": "Corporate Disclosures"},
        {"text": "Corporate governance committee clears Managing Director of any immediate conflict of interest.", "true_sentiment": "positive", "category": "Corporate Disclosures"},

        # --- BLIND SPOT 3: INDIAN BUSINESS PRESS METAPHORS ---
        {"text": "Rural consumption under heavy pressure as monsoon deficit expands to twelve percent.", "true_sentiment": "negative", "category": "Media Idioms"},
        {"text": "Promoter share pledge reaches dangerous thresholds of forty five percent for utility group.", "true_sentiment": "negative", "category": "Media Idioms"},
        {"text": "Automobile dealerships report explosive festive demand pickup across tier two cities.", "true_sentiment": "positive", "category": "Media Idioms"},
        {"text": "Dalal Street indices snap gaining streak, ending down on heavy institutional profit booking.", "true_sentiment": "negative", "category": "Media Idioms"},
        {"text": "Midcap stocks witness intense bloodbath as leverage unwinding triggers margin calls.", "true_sentiment": "negative", "category": "Media Idioms"},
        {"text": "DII flows continue to provide strong safety cushion against persistent FII selling pressure.", "true_sentiment": "positive", "category": "Media Idioms"},
        {"text": "Corporate earnings season kicks off on a tepid note; operational margins compressed.", "true_sentiment": "negative", "category": "Media Idioms"},
        {"text": "FMCG volumes show early signs of stabilizing after prolonged rural slowdown pain.", "true_sentiment": "positive", "category": "Media Idioms"},
        {"text": "Sustained capex push by public sector units sets up structural multi year infra rally.", "true_sentiment": "positive", "category": "Media Idioms"},
        {"text": "Commodity inflation risks cooling off, handing structural margin relief to manufacturing units.", "true_sentiment": "positive", "category": "Media Idioms"},
        {"text": "SME IPO segment shows signs of classic overheating, forcing closer valuation checks.", "true_sentiment": "negative", "category": "Media Idioms"},
        {"text": "Tax collection run rate outpaces budgetary estimates, offering deep fiscal comfort.", "true_sentiment": "positive", "category": "Media Idioms"},

        # --- BLIND SPOT 4: HINDI-ENGLISH CODE-SWITCHING (HINGLISH) ---
        {"text": "Nifty options me heavy bikwali happening today as global markets crack.", "true_sentiment": "negative", "category": "Hinglish Retail"},
        {"text": "Market looks fully bullish, targets of 25k coming soon boys.", "true_sentiment": "positive", "category": "Hinglish Retail"},
        {"text": "Loss booking start karo, market is looking highly dangerous right now.", "true_sentiment": "negative", "category": "Hinglish Retail"},
        {"text": "Retail investors ka confidence solid hai despite multiple minor correction dips.", "true_sentiment": "positive", "category": "Hinglish Retail"},
        {"text": "Portfolio completely red ho gaya, zero liquidity left to buy structural dips.", "true_sentiment": "negative", "category": "Hinglish Retail"},
        {"text": "Operators are trapping retail public in penny stocks, trading call safely.", "true_sentiment": "negative", "category": "Hinglish Retail"},
        {"text": "Earnings figures are clean, big dhamaka breakout coming post results announcement.", "true_sentiment": "positive", "category": "Hinglish Retail"},
        {"text": "FII exit ho gaye, now local institutions will drive the absolute target rally.", "true_sentiment": "positive", "category": "Hinglish Retail"},
        {"text": "SIP flows har month badh rahe hain, market crash easily absorb ho jayega.", "true_sentiment": "positive", "category": "Hinglish Retail"},
        {"text": "Market sideways chal raha hai, clear breakout direction ka wait karo.", "true_sentiment": "neutral", "category": "Hinglish Retail"},
        {"text": "Result bakwaas hai, stock prices will crash hard on morning trade opening.", "true_sentiment": "negative", "category": "Hinglish Retail"},
        {"text": "Smallcaps completely dump ho rahe hain, retail public running away completely.", "true_sentiment": "negative", "category": "Hinglish Retail"},
        {"text": "Budget proposals positive hain, sector long term investment looks rock solid.", "true_sentiment": "positive", "category": "Hinglish Retail"},
        {"text": "F&O positions me heavy traps ready, do not short the current baseline support.", "true_sentiment": "neutral", "category": "Hinglish Retail"}
    ]

def main():
    print("Initializing Western FinBERT (ProsusAI/finbert)...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    # FinBERT label map tracking
    finbert_labels = {0: "positive", 1: "negative", 2: "neutral"}
    
    data = load_audit_dataset()
    results = []

    print(# Running inference loop over all 50 headlines...
    "\nExecuting Inference Audit on 50 Indian Market Headlines...")
    
    with torch.no_grad():
        for item in data:
            inputs = tokenizer(item["text"], return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            
            # Use softmax to evaluate internal model confidence metrics
            probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = finbert_labels[pred_idx]
            
            results.append({
                "Text": item["text"],
                "Category": item["category"],
                "True_Label": item["true_sentiment"],
                "Predicted_Label": pred_label,
                "Confidence": probs[pred_idx],
                "Is_Correct": 1 if pred_label == item["true_sentiment"] else 0
            })

    df = pd.DataFrame(results)
    
    # Global Performance Calculations
    total_accuracy = accuracy_score(df["True_Label"], df["Predicted_Label"])
    error_count = len(df) - df["Is_Correct"].sum()

    print("\n" + "="*50)
    print("          FINBERT GLOBAL RISK METRICS REPORT       ")
    print("="*50)
    print(f"Total Sample Space Evaluated : {len(df)} Local Headlines")
    print(f"Empirical Model Accuracy     : {total_accuracy * 100:.1f}%")
    print(f"Critical Prediction Failures : {error_count} Out of 50 Sequences")
    print("="*50)

    print("\n--- Structural Blind Spot Category Breakdown ---")
    category_summary = df.groupby("Category")["Is_Correct"].agg(
        Total_Samples="count",
        Correct_Predictions="sum",
        Accuracy=lambda x: f"{(x.sum()/x.count())*100:.1f}%"
    )
    print(category_summary)

    print("\n--- Top 5 Critical Operational Failures (OOD Gaps) ---")
    failures = df[df["Is_Correct"] == 0].head(5)
    for idx, row in failures.iterrows():
        print(f"\n[BLIND SPOT: {row['Category']}]")
        print(f"Headline : '{row['Text']}'")
        print(f"Reality  : [{row['True_Label'].upper()}]  -->  FinBERT output: [{row['Predicted_Label'].upper()}] (Confidence: {row['Confidence']*100:.1f}%)")

if __name__ == "__main__":
    main()