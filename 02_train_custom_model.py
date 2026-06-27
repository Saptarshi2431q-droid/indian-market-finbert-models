"""
Step 2: Train AI4Invest Custom Model (Clean Data)
"""
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import warnings
warnings.filterwarnings("ignore")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc}

def main():
    print("IGNITING CUSTOM FINE-TUNING PIPELINE...")
    
    # Authenticate WandB
    wandb.login(key="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")
    wandb.init(project="AI4Invest-FinBERT-Tuning", name="Production_Run_Clean_Data")

    # 1. Load the NEW Unbiased Data
    try:
        df = pd.read_csv("unbiased_financial_headlines.csv")
    except Exception as e:
        print("[ERROR] Cannot find 'unbiased_financial_headlines.csv'. Run script 00 first.")
        return

    label_dict = {"Bullish": 0, "Bearish": 1, "Neutral": 2}
    df['label'] = df['Predicted_Sentiment'].map(label_dict)
    df = df[['Headline', 'label']]

    # 2. Split Data (80% Train, 20% Holdout)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    # 3. Tokenize
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    def tokenize_function(examples):
        return tokenizer(examples["Headline"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 4. Initialize Base Model
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    
    training_args = TrainingArguments(
        output_dir="./indian_finbert_model",
        report_to="wandb",
        evaluation_strategy="epoch",      
        learning_rate=2e-5,               
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        num_train_epochs=3,               
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    # 5. Execute Training
    trainer.train()

    # 6. Save Custom Model
    print("\nSaving Custom Model...")
    trainer.save_model("./models/optimized_indian_finbert_final")
    tokenizer.save_pretrained("./models/optimized_indian_finbert_final")
    wandb.finish()
    print("[SUCCESS] Training complete and model saved.")

if __name__ == "__main__":
    main()