"""
Layer 11: Production Fine-Tuning of FinBERT
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Mathematically update BERT weights using backpropagation on Dalal Street corpus.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
# INJECTED: Added precision_recall_fscore_support and confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    """Calculates mathematical evaluation metrics during training.""" 
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # INJECTED: Calculating Precision, Recall, and F1 simultaneously
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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

    # 2. Map String Labels to Integers (BERT only understands math, not text)
    label_dict = {"Bullish": 0, "Bearish": 1, "Neutral": 2}
    df['label'] = df['Predicted_Sentiment'].map(label_dict)
    
    # Clean the data: keep only what we need
    df = df[['Headline', 'label']]

    # 3. Train/Test Split (80% for learning, 20% for evaluating)
    print("\nSplitting data into Training (80%) and Evaluation (20%) sets...")
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Convert Pandas DataFrames into HuggingFace Datasets
    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    # 4. Tokenization (Converting words to vectors)
    print("\nLoading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    def tokenize_function(examples):
        return tokenizer(examples["Headline"], padding="max_length", truncation=True, max_length=64)

    print("Tokenizing datasets (This prepares the vectors for backpropagation)...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 5. Load the Base Model
    print("\nLoading Base Model for Weight Updates...")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Hardware Detected: {device.type.upper()}")
    
    if device.type == 'cpu':
        print("\n[WARNING] You are training on a CPU. Fine-tuning neural networks on a CPU")
        print("can take a long time. We have reduced the training limits to compensate.")

    # 6. Define Training Arguments (The hyperparameter setup)
    print("\nConfiguring Backpropagation Parameters...")
    training_args = TrainingArguments(
        output_dir="./indian_finbert_model",
        evaluation_strategy="epoch",      # Evaluate at the end of each epoch
        learning_rate=2e-5,               # Standard learning rate for fine-tuning
        per_device_train_batch_size=16,   # Batch size for memory safety
        per_device_eval_batch_size=16,
        num_train_epochs=3,               # Pass through the data 3 times
        weight_decay=0.01,                # Prevent overfitting
        logging_dir='./logs',
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,      # Automatically keep the best version
    )

    # 7. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    # 8. THE ACTUAL FINE-TUNING
    print("\n==================================================")
    print("  INITIATING GRADIENT DESCENT & FINE-TUNING")
    print("==================================================")
    print("Please wait. This process alters the neural network weights.")
    
    trainer.train()

    # 9. Save the Final Model
    print("\n[SUCCESS] Fine-tuning complete!")
    print("Saving the new 'Indian FinBERT' model to your local drive...")
    trainer.save_model("./optimized_indian_finbert_final")
    tokenizer.save_pretrained("./optimized_indian_finbert_final")
    
    # 10. Final Evaluation
    print("\nRunning final evaluation on the 20% holdout set...")
    eval_results = trainer.evaluate()
    
    # INJECTED: Added Precision, Recall, and Confusion Matrix logic
    print("\n--- FINAL MODEL REPORT ---")
    print(f"Accuracy  : {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Precision : {eval_results['eval_precision']:.4f}")
    print(f"Recall    : {eval_results['eval_recall']:.4f}")
    print(f"F1-Score  : {eval_results['eval_f1']:.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    # Forcing the model to predict the evaluation set to generate the matrix
    predictions = trainer.predict(tokenized_eval)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    
    print("         Predicted")
    print("         0   1   2")
    print("       +---+---+---+")
    print(f"Actual 0| {cm[0][0]:>2} | {cm[0][1]:>2} | {cm[0][2]:>2} | (Bullish)")
    print(f"       1| {cm[1][0]:>2} | {cm[1][1]:>2} | {cm[1][2]:>2} | (Bearish)")
    print(f"       2| {cm[2][0]:>2} | {cm[2][1]:>2} | {cm[2][2]:>2} | (Neutral)")
    print("       +---+---+---+")

    print("\nModel successfully saved in folder: ./optimized_indian_finbert_final")

if __name__ == "__main__":
    main()