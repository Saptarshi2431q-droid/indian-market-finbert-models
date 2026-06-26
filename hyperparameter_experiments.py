"""
Layer 11: Experimental Hyperparameter Tuning
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Live MLOps tracking of hyperparameters and extended evaluation metrics.
"""

import pandas as pd
import numpy as np
import torch
import wandb
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

# ==========================================
# ⚙️ HYPERPARAMETER CONTROL CENTER ⚙️
# Change these values for each new experiment!
# ==========================================
CURRENT_LEARNING_RATE = 2e-5
CURRENT_EPOCHS = 3
CURRENT_WEIGHT_DECAY = 0.001

# This automatically names your graph on the website so you know which experiment it is
RUN_NAME = f"LR-{CURRENT_LEARNING_RATE}_EP-{CURRENT_EPOCHS}_WD-{CURRENT_WEIGHT_DECAY}"

def compute_metrics(pred):
    """Calculates all 5 mathematical metrics, including MCC.""" 
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }

def main():
    print("==================================================")
    print(f"  IGNITING EXPERIMENT: {RUN_NAME}")
    print("==================================================")
    
    # 1. Hardcoded API Authentication (Bypassing Windows Security)
    wandb.login(key="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")
    
    # 2. Ignite the Weights & Biases Live Tracker
    wandb.init(project="AI4Invest-FinBERT-Tuning", name=RUN_NAME)

    # 3. Data Loading & Prep
    try:
        df = pd.read_csv("classified_historical_headlines.csv")
    except Exception as e:
        print("[ERROR] Could not load dataset.")
        return

    label_dict = {"Bullish": 0, "Bearish": 1, "Neutral": 2}
    df['label'] = df['Predicted_Sentiment'].map(label_dict)
    df = df[['Headline', 'label']]

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    def tokenize_function(examples):
        return tokenizer(examples["Headline"], padding="max_length", truncation=True, max_length=64)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    
    # 4. Configure TrainingArguments to broadcast to WandB
    training_args = TrainingArguments(
        output_dir="./experimental_finbert_models", 
        report_to="wandb",                
        run_name=RUN_NAME,
        evaluation_strategy="epoch",      
        learning_rate=CURRENT_LEARNING_RATE,               
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        num_train_epochs=CURRENT_EPOCHS,               
        weight_decay=CURRENT_WEIGHT_DECAY,                
        logging_dir='./logs',
        logging_steps=10,                 
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
    
    # 5. Execute
    trainer.train()

    print("\nRunning final evaluation on the 20% holdout set...")
    eval_results = trainer.evaluate()
    
    # 6. Send the final 3x3 Confusion Matrix to the dashboard
    predictions = trainer.predict(tokenized_eval)
    preds = np.argmax(predictions.predictions, axis=-1)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=predictions.label_ids, preds=preds,
                            class_names=["Bullish", "Bearish", "Neutral"])})
    
    wandb.finish()
    print("\n[SUCCESS] Experiment complete and telemetry uploaded to WandB!")

if __name__ == "__main__":
    main()