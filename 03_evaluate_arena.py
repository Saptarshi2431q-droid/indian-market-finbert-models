import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

def evaluate_and_diagnose(model_name, model_path, texts, true_labels):
    print(f"\n==================================================")
    print(f"  [DIAGNOSTICS RUNNING] {model_name}")
    print(f"==================================================")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Determine how this specific model maps its internal math to English labels
        id2label = model.config.id2label
        
        # We need to map the model's output back to our AI4Invest standard:
        # 0: Bullish, 1: Bearish, 2: Neutral
        standard_map = {"positive": 0, "bullish": 0, "negative": 1, "bearish": 1, "neutral": 2}
        
        preds = []
        # Process in small batches to save memory
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=-1).numpy()
                
                for p in batch_preds:
                    # Translate the model's specific label into standard English, then to our integers
                    english_label = id2label[p].lower()
                    standard_int = standard_map.get(english_label, 2) # Default to neutral if weird
                    preds.append(standard_int)
                    
        cm = confusion_matrix(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(true_labels, preds)
        mcc = matthews_corrcoef(true_labels, preds)
        
        metrics = {'test_accuracy': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1, 'test_mcc': mcc}
        return metrics, cm
    except Exception as e:
        print(f"[ERROR] Could not process {model_path}. Error: {e}")
        return None, None

def main():
    print("\n==================================================")
    print("  AI4INVEST: THE GLADIATORIAL ARENA")
    print("==================================================")
    
    try:
        df = pd.read_csv("unbiased_financial_headlines.csv")
    except:
        print("[ERROR] Cannot find 'unbiased_financial_headlines.csv'.")
        return

    label_dict = {"Bullish": 0, "Bearish": 1, "Neutral": 2}
    df['label'] = df['Predicted_Sentiment'].map(label_dict)
    
    _, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    texts = eval_df['Headline'].tolist()
    true_labels = eval_df['label'].tolist()

    competitors = {
        "Western FinBERT (ProsusAI)": "./models/western_finbert_baseline",
        "Indian FinBERT (Vansh180)": "./models/hf_indian_finbert_baseline",
        "Rishi's AI4Invest FinBERT": "./models/optimized_indian_finbert_final"
    }
    
    scoreboard = {}
    for name, path in competitors.items():
        metrics, cm = evaluate_and_diagnose(name, path, texts, true_labels)
        
        if metrics:
            scoreboard[name] = metrics
            print(f"\n--- {name.upper()} TELEMETRY ---")
            print(f"Accuracy  : {metrics['test_accuracy']:.4f}")
            print(f"Precision : {metrics['test_precision']:.4f}")
            print(f"Recall    : {metrics['test_recall']:.4f}")
            print(f"F1-Score  : {metrics['test_f1']:.4f}")
            print(f"MCC       : {metrics['test_mcc']:.4f}")
            
            print("\n--- CONFUSION MATRIX ---")
            print("          Predicted")
            print("          0   1   2")
            print("        +---+---+---+")
            print(f"Actual 0| {cm[0][0]:>3} | {cm[0][1]:>3} | {cm[0][2]:>3} | (Bullish)")
            print(f"       1| {cm[1][0]:>3} | {cm[1][1]:>3} | {cm[1][2]:>3} | (Bearish)")
            print(f"       2| {cm[2][0]:>3} | {cm[2][1]:>3} | {cm[2][2]:>3} | (Neutral)")
            print("        +---+---+---+\n")
            
    print("\n=================================================================")
    print(" FINAL SCOREBOARD (HOLDOUT SET EVALUATION)")
    print("=================================================================")
    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'MCC':<10}")
    print("-" * 65)
    
    for name, scores in scoreboard.items():
        acc = round(scores['test_accuracy'], 4)
        f1 = round(scores['test_f1'], 4)
        mcc = round(scores['test_mcc'], 4)
        print(f"{name:<30} | {acc:<10} | {f1:<10} | {mcc:<10}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()