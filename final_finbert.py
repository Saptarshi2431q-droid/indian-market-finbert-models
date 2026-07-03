"""
Project: AI4Invest-Indian-FinBERT MLOps Pipeline
Layer: Production Fine-Tuning Script (Phase 2 SOTA Implementation)
Author: Saptarshi Dutta (Rishi)
Target Review: Durga Bhaiyaa / XLRI Jamshedpur Admissions Panel
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datasets
import wandb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    get_cosine_schedule_with_warmup
)

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ADVANCED TEXT PREPROCESSING LOGIC
# ==============================================================================
def clean_financial_tweet(text):
    """
    Strips noise from retail market chatter to clean the embedding space.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)                   # Remove Twitter Handles
    text = re.sub(r'\$\w+', '', text)                  # Remove Cashtags ($RELIANCE)
    text = re.sub(r'\s+', ' ', text).strip()           # Clean whitespace
    return text

# ==============================================================================
# 2. CUSTOM LOSS OVERSIGHT (CLASS WEIGHTING)
# ==============================================================================
class ImbalanceMitigationTrainer(Trainer):
    """
    Custom Hugging Face Trainer that overrides the loss computation engine.
    Injects normalized inverse-frequency class weights into CrossEntropyLoss.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # **kwargs safely absorbs parameters like num_items_in_batch from newer HF versions
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None and labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# 3. COMPREHENSIVE INSTITUTIONAL METRICS EVALUATOR
# ==============================================================================
def compute_metrics(pred):
    """
    Calculates operational metrics to meet enterprise desk mandates.
    """
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

# ==============================================================================
# 4. ORCHESTRATION PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print("IGNITING PRODUCTION-GRADE FINE-TUNING EXECUTION PIPELINE")
    print("=" * 60)

    # Telemetry Check
    if "WANDB_API_KEY" not in os.environ:
        print("[WARNING] WANDB_API_KEY environment variable not found. Fallback to local tracking or manual prompt.")
    
    try:
        wandb.init(project="AI4Invest-FinBERT-Tuning", name="Production_Run_Phase2_Optimized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Weights & Biases: {e}")
        print("Proceeding with local training tracking only.")

    # Data Ingestion & Sanitization
    data_path = "unbiased_financial_headlines.csv" 
    print(f"\n[1/6] Ingesting dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[ERROR] Mandated file '{data_path}' missing from workspace root.")
        sys.exit(1)

    # Standardizing incoming data structures
    text_col = 'Headline' if 'Headline' in df.columns else df.columns[0]
    label_col = 'Predicted_Sentiment' if 'Predicted_Sentiment' in df.columns else 'label'

    # STRICT LABEL PARSER: Enforces Integer output regardless of input format (handles "0", "bullish", 0)
    def parse_label(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return int(val)
        
        val_str = str(val).strip().lower()
        label_map = {'bullish': 0, 'bearish': 1, 'neutral': 2}
        
        if val_str in label_map:
            return label_map[val_str]
        elif val_str.isdigit():
            return int(val_str)
        else:
            return np.nan

    df['label'] = df[label_col].apply(parse_label)
    
    # Drop unmatched records and firmly cast to integer before hitting PyTorch
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    
    print(f"Applying Twitter Text Preprocessing to {len(df)} records...")
    df['cleaned_text'] = df[text_col].astype(str).apply(clean_financial_tweet)

    # Stratified Dataset Splitting
    print("\n[2/6] Generating stratified 80/20 train/validation partitions...")
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Calculate Class Weights to pass to our Custom Loss function
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = len(train_df)
    calculated_weights = total_samples / (3.0 * class_counts)
    print(f"Calculated Inverse Frequency Class Weights: {calculated_weights}")

    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    # Tokenization Vectors Generation
    print("\n[3/6] Fetching ProsusAI/finbert Tokenizer & encoding text spaces...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # Hardware & Model Initialization
    print("\n[4/6] Instantiating neural network architecture weights...")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Hardware: {device.type.upper()}")

    # Training Configuration Matrix
    print("\n[5/6] Tuning training hyperparameters and optimization schedules...")
    training_args = TrainingArguments(
        output_dir="./indian_finbert_model",
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        eval_strategy="epoch",      
        learning_rate=2e-5,               
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        num_train_epochs=3,               
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        logging_steps=25
    )

    # Build custom trainer injected with our Phase 2 enhancements
    trainer = ImbalanceMitigationTrainer(
        class_weights=calculated_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    # Custom LRScheduler Setup (Cosine Annealing with Warmup)
    num_training_steps = len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    num_warmup_steps = int(0.10 * num_training_steps) # 10% Warmup phase
    
    trainer.create_optimizer_and_scheduler(num_training_steps=num_training_steps)
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=trainer.optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Backpropagation execution block
    print("\n[6/6] COMMENCING GRADIENT DESCENT OPTIMIZATION RUN...")
    trainer.train()

    # IO Persist Operations
    print("\nSaving Optimized Custom Model Checkpoints...")
    output_directory = "./models/optimized_indian_finbert_final"
    trainer.save_model(output_directory)
    tokenizer.save_pretrained(output_directory)
    
    # Holdout Verification Diagnostics
    print("\n" + "="*40 + "\nFINAL MODEL PRODUCTION REPORT\n" + "="*40)
    eval_results = trainer.evaluate()
    print(f"Operational Accuracy  : {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Precision Score       : {eval_results['eval_precision']:.4f}")
    print(f"Recall Score          : {eval_results['eval_recall']:.4f}")
    print(f"Institutional F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"Matthews Corr. (MCC)  : {eval_results['eval_mcc']:.4f}")

    # Generate Confusion Matrix
    predictions = trainer.predict(tokenized_eval)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    
    print("\n--- CONFUSION MATRIX RE-VERIFICATION ---")
    print("                 Predicted Labels")
    print("                 Bullish  Bearish  Neutral")
    print("               +---------+--------+---------+")
    print(f"Actual Bullish |   {cm[0][0]:>4}  |  {cm[0][1]:>4}  |   {cm[0][2]:>4}  |")
    print(f"Actual Bearish |   {cm[1][0]:>4}  |  {cm[1][1]:>4}  |   {cm[1][2]:>4}  |")
    print(f"Actual Neutral |   {cm[2][0]:>4}  |  {cm[2][1]:>4}  |   {cm[2][2]:>4}  |")
    print("               +---------+--------+---------+")
    
    try:
        wandb.finish()
    except Exception:
        pass
        
    print(f"\n[SUCCESS] Custom Pipeline Finished. Deployment artifacts located at: {output_directory}")

if __name__ == "__main__":
    main()