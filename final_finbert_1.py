"""
Project: AI4Invest-Indian-FinBERT MLOps Pipeline
Layer: Production Fine-Tuning Script (Ultimate Architecture)
Author: Saptarshi Dutta (Rishi)
Architecture Upgrades: Weighted Label Smoothing, LLRD (Layer-Wise LR Decay), Dynamic Padding
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datasets
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. TEXT PREPROCESSING
# ==============================================================================
def clean_financial_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\$\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# 2. LABEL SMOOTHING + CLASS WEIGHTS ENGINE
# ==============================================================================
class SmoothWeightTrainer(Trainer):
    """
    Combines Inverse Class Weights with Label Smoothing.
    Prevents the model from being overconfident on noisy, ambiguous financial labels.
    """
    def __init__(self, class_weights=None, smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Apply Label Smoothing manually with class weights
        num_classes = logits.size(-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
            
        if self.class_weights is not None:
            # Scale the true distribution by our class weights
            weight_gather = self.class_weights[labels].unsqueeze(1)
            loss = (-true_dist * log_probs).sum(dim=-1) * weight_gather.squeeze()
            loss = loss.mean()
        else:
            loss = (-true_dist * log_probs).sum(dim=-1).mean()
            
        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# 3. METRICS EVALUATOR
# ==============================================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc}

# ==============================================================================
# 4. ORCHESTRATION PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("IGNITING ULTIMATE MLOPS ARCHITECTURE (SMOOTHING + LLRD)")
    print("=" * 70)

    data_path = "unbiased_financial_headlines.csv" 
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[ERROR] '{data_path}' missing.")
        sys.exit(1)

    text_col = 'Headline' if 'Headline' in df.columns else df.columns[0]
    label_col = 'Predicted_Sentiment' if 'Predicted_Sentiment' in df.columns else 'label'

    def parse_label(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return int(val)
        val_str = str(val).strip().lower()
        label_map = {'bullish': 0, 'bearish': 1, 'neutral': 2}
        if val_str in label_map: return label_map[val_str]
        elif val_str.isdigit(): return int(val_str)
        return np.nan

    df['label'] = df[label_col].apply(parse_label)
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    
    print(f"Sanitizing {len(df)} records...")
    df['cleaned_text'] = df[text_col].astype(str).apply(clean_financial_tweet)

    train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    
    # Calculate Weights
    class_counts = train_df['label'].value_counts().sort_index().values
    calculated_weights = len(train_df) / (3.0 * class_counts)
    
    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], truncation=True, max_length=128)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["cleaned_text", text_col])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["cleaned_text", text_col])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)

    # ---------------------------------------------------------
    # LAYER-WISE LEARNING RATE DECAY (LLRD)
    # ---------------------------------------------------------
    optimizer_grouped_parameters = []
    base_lr = 2e-5
    decay_factor = 0.95
    
    # Classification head gets highest LR
    optimizer_grouped_parameters.append(
        {"params": model.classifier.parameters(), "lr": base_lr, "weight_decay": 0.01}
    )
    # 12 Encoder layers get progressively lower LRs
    for i in range(11, -1, -1):
        layer_lr = base_lr * (decay_factor ** (12 - i))
        optimizer_grouped_parameters.append(
            {"params": model.bert.encoder.layer[i].parameters(), "lr": layer_lr, "weight_decay": 0.01}
        )
    # Embeddings get the lowest LR
    optimizer_grouped_parameters.append(
        {"params": model.bert.embeddings.parameters(), "lr": base_lr * (decay_factor ** 13), "weight_decay": 0.0}
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    training_args = TrainingArguments(
        output_dir="./indian_finbert_model",
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        fp16=True if torch.cuda.is_available() else False,
        num_train_epochs=4,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True
    )

    trainer = SmoothWeightTrainer(
        class_weights=calculated_weights,
        smoothing=0.1,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        optimizers=(optimizer, None) # Pass our custom LLRD optimizer
    )

    num_training_steps = len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.10 * num_training_steps),
        num_training_steps=num_training_steps
    )

    print("\n[6/6] COMMENCING OPTIMIZATION...")
    trainer.train()

    print("\n" + "="*40 + "\nFINAL PRODUCTION REPORT\n" + "="*40)
    eval_results = trainer.evaluate()
    print(f"Operational Accuracy  : {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Institutional F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"Matthews Corr. (MCC)  : {eval_results['eval_mcc']:.4f}")

    predictions = trainer.predict(tokenized_eval)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"Actual Bullish |   {cm[0][0]:>4}  |  {cm[0][1]:>4}  |   {cm[0][2]:>4}  |")
    print(f"Actual Bearish |   {cm[1][0]:>4}  |  {cm[1][1]:>4}  |   {cm[1][2]:>4}  |")
    print(f"Actual Neutral |   {cm[2][0]:>4}  |  {cm[2][1]:>4}  |   {cm[2][2]:>4}  |")

if __name__ == "__main__":
    main()