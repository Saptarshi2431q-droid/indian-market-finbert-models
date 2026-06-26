"""
Transformer Core Sandbox - Multi-Head Attention from Scratch
Author: Technical Internship Portfolio
Framework: PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        num_odd_cols = pe[:, 1::2].size(1)
        pe[:, 1::2] = torch.cos(position * div_term[:num_odd_cols])
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be perfectly divisible by num_heads!"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension size for each individual specialist head
        
        # Linear projection layers to create Q, K, and V spaces
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection layer to unify all head inputs at the end
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Step 1: Project input into Q, K, V matrices
        # Shape shifts from: [Batch, Seq_Len, d_model] -> [Batch, Seq_Len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Step 2: Split matrices into multiple heads using tensor view manipulation
        # New Target Shape: [Batch, Num_Heads, Seq_Len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Compute Scaled Dot-Product Attention for all heads simultaneously
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(raw_scores, dim=-1)
        head_outputs = torch.matmul(attention_weights, V)
        
        # Step 4: Concatenate ("glue") all the head outputs back together
        # Shape shifts back: [Batch, Num_Heads, Seq_Len, d_k] -> [Batch, Seq_Len, d_model]
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Step 5: Run through the final linear transformation layer
        final_output = self.w_out(concat_output)
        
        return final_output, attention_weights


# ==========================================
# EXECUTION & PIPELINE INSPECTION
# ==========================================
if __name__ == "__main__":
    # Let's set up a system with 4 features and 2 distinct attention heads
    features_dim = 4 
    heads_count = 2
    
    pos_encoder = PositionalEncoding(d_model=features_dim, max_len=10)
    mha_layer = MultiHeadAttention(d_model=features_dim, num_heads=heads_count)

    # Simulated market data: [Batch=1, Days=4, Features=4]
    # Features: [Return, Volume, Volatility, Moving_Average_Delta]
    raw_market_data = torch.tensor([
        [[0.5, 0.1, -0.2, 0.8],   # Day 1
         [0.6, 0.2, -0.1, 0.7],   # Day 2
         [-1.5, 3.2, 2.5, -0.9],  # Day 3 (Extreme Outlier/Crash Day)
         [0.4, 0.1, -0.3, 0.6]]   # Day 4 (Today)
    ], dtype=torch.float32)

    # Run through the pipeline
    time_aware_data = pos_encoder(raw_market_data)
    output_features, weights = mha_layer(time_aware_data, time_aware_data, time_aware_data)

    print("--- MULTI-HEAD ATTENTION METRICS ---")
    print("Output Features Shape:", output_features.shape)
    print("Attention Weights Shape:", weights.shape) # [Batch, Heads, Seq_Len, Seq_Len]

    # Inspect what each head focused on for Day 4 (Today)
    for h in range(heads_count):
        head_weights_day4 = torch.round(weights[0, h, 3] * 100)
        print(f"\nSpecialist Head {h+1} Attention Distribution for Today (Day 4):")
        print(f"-> Day 1: {head_weights_day4[0].item()}%, "
              f"Day 2: {head_weights_day4[1].item()}%, "
              f"Day 3: {head_weights_day4[2].item()}%, "
              f"Day 4: {head_weights_day4[3].item()}%")