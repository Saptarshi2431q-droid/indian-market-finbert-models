import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def prepare_data():
    print("Downloading Nifty 50 data...")
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(start="2018-01-01")
    
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    
    num_lags = 10
    for i in range(1, num_lags + 1):
        df[f'Lag_{i}'] = df['Return'].shift(i)
        
    df = df.dropna()
    
    feature_cols = [f'Lag_{i}' for i in range(1, num_lags + 1)]
    X = df[feature_cols].values
    y = df['Return'].values.reshape(-1, 1)
    
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

class NiftyMLP(nn.Module):
    def __init__(self, input_dim):
        super(NiftyMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.out(x)
        return x

def main():
    X_train, X_test, y_train, y_test = prepare_data()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = NiftyMLP(input_dim=10)
    
    # --- CRITICAL CHANGE: Using Huber Loss instead of MSE ---
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 50
    print("\nStarting MLP training loop with Huber Loss...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        total_epoch_loss = epoch_loss / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Huber Loss: {total_epoch_loss:.6f}")
            
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        test_loss = criterion(test_predictions, y_test_t)
        print(f"\nMLP Model Training Complete.")
        print(f"Final Out-of-Sample Test Huber Loss: {test_loss.item():.6f}")

if __name__ == "__main__":
    main()