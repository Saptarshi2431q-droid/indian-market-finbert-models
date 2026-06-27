"""
Layer 11: Production Fine-Tuning & Competitor Evaluation
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Evaluate Hugging Face baselines, then fine-tune custom BERT weights.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import datasets
import warnings
warnings.filterwarnings("ignore")

def compute_metrics(pred):
    """Calculates mathematical evaluation metrics, including MCC.""" 
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds) # INJECTED: MCC Logic
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }

def evaluate_baseline(repo_id, tokenized_eval_dataset):
    """Helper function to download and grade competitor models without training them."""
    print(f"\n[DOWNLOADING & EVALUATING BASELINE] {repo_id}...")
    try:
        # This automatically downloads the weights from Hugging Face
        model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=3)
        eval_args = TrainingArguments(output_dir="./temp_eval", per_device_eval_batch_size=16, report_to="none")
        
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=compute_metrics,
        )
        return trainer.evaluate()
    except Exception as e:
        print(f"[ERROR] Could not process {repo_id}. It will be skipped.")
        return None

def main():
    print("==================================================")
    print("  INITIALIZING INDIAN FINBERT FINE-TUNING PIPELINE")
    print("==================================================")
    
    # 1. Load the Data
    try:
        df = pd.read_csv("classified_historical_headlines.csv")
        print(f"Loaded {len(df)} headlines.")
    except Exception as e:
        print("[ERROR] Could not load dataset. Make sure the CSV is in the same folder.")
        return

    # 2. Map String Labels to Integers
    label_dict = {"Bullish": 0, "Bearish": 1, "Neutral": 2}
    df['label'] = df['Predicted_Sentiment'].map(label_dict)
    df = df[['Headline', 'label']]

    # 3. Train/Test Split
    print("\nSplitting data into Training (80%) and Evaluation (20%) sets...")
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    # 4. Tokenization
    print("\nLoading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    def tokenize_function(examples):
        return tokenizer(examples["Headline"], padding="max_length", truncation=True, max_length=64)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # ==================================================
    # PHASE 1: EVALUATE THE COMPETITION FIRST
    # ==================================================
    print("\n==================================================")
    print("  PHASE 1: BENCHMARKING HUGGING FACE BASELINES")
    print("==================================================")
    
    western_repo = "ProsusAI/finbert"
    indian_repo = "yashkant/finbert-indian-stock-market"
    
    western_scores = evaluate_baseline(western_repo, tokenized_eval)
    indian_scores = evaluate_baseline(indian_repo, tokenized_eval)

    # ==================================================
    # PHASE 2: TRAIN YOUR CUSTOM MODEL
    # ==================================================
    print("\n==================================================")
    print("  PHASE 2: INITIATING GRADIENT DESCENT & FINE-TUNING")
    print("==================================================")
    
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    
    training_args = TrainingArguments(
        output_dir="./indian_finbert_model",
        evaluation_strategy="epoch",      
        learning_rate=2e-5,               
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        num_train_epochs=3,               
        weight_decay=0.01,                
        logging_dir='./logs',
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none" # Keeping terminal clean
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nSaving the new 'Indian FinBERT' model to your local drive...")
    trainer.save_model("./optimized_indian_finbert_final")
    tokenizer.save_pretrained("./optimized_indian_finbert_final")
    
    print("\nRunning final evaluation on your custom model...")
    custom_scores = trainer.evaluate()

    # ==================================================
    # PHASE 3: THE FINAL REPORT
    # ==================================================
    print("\n=================================================================")
    print("  FINAL COMPETITOR SCOREBOARD (INCLUDES MCC)")
    print("=================================================================")
    print(f"{'Model Architecture':<35} | {'Accuracy':<10} | {'F1-Score':<10} | {'MCC':<10}")
    print("-" * 72)
    
    def print_row(name, scores):
        if scores:
            acc = f"{scores['eval_accuracy']:.4f}"
            f1 = f"{scores['eval_f1']:.4f}"
            mcc = f"{scores['eval_mcc']:.4f}"
            print(f"{name:<35} | {acc:<10} | {f1:<10} | {mcc:<10}")

    print_row("Western FinBERT (ProsusAI)", western_scores)
    print_row("Generic Indian FinBERT (Yashkant)", indian_scores)
    print_row("Rishi's AI4Invest FinBERT", custom_scores)
    print("=================================================================\n")

    print("\n--- CONFUSION MATRIX (CUSTOM MODEL) ---")
    predictions = trainer.predict(tokenized_eval)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    
    print("        Predicted")
    print("        0   1   2")
    print("      +---+---+---+")
    print(f"Actual 0| {cm[0][0]:>2} | {cm[0][1]:>2} | {cm[0][2]:>2} | (Bullish)")
    print(f"       1| {cm[1][0]:>2} | {cm[1][1]:>2} | {cm[1][2]:>2} | (Bearish)")
    print(f"       2| {cm[2][0]:>2} | {cm[2][1]:>2} | {cm[2][2]:>2} | (Neutral)")
    print("      +---+---+---+")

if __name__ == "__main__":
    main()