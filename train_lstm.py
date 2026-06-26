import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def prepare_lstm_data():
    print("Downloading Nifty 50 data...")
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(start="2018-01-01")
    
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    
    num_lags = 10
    for i in range(1, num_lags + 1):
        df[f'Lag_{i}'] = df['Return'].shift(i)
        
    df = df.dropna()
    
    feature_cols = [f'Lag_{i}' for i in range(num_lags, 0, -1)] 
    X = df[feature_cols].values
    y = df['Return'].values.reshape(-1, 1)
    
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = np.reshape(X_train, (X_train.shape[0], num_lags, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], num_lags, 1))
    
    return X_train, X_test, y_train, y_test

class NiftyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(NiftyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :] 
        prediction = self.out(last_time_step_out)
        return prediction

def main():
    X_train, X_test, y_train, y_test = prepare_lstm_data()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = NiftyLSTM(input_size=1, hidden_size=64, num_layers=1, output_size=1)
    
    # --- CRITICAL CHANGE: Using Huber Loss instead of MSE ---
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 50
    print("\nStarting LSTM training loop with Huber Loss...")
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
        print(f"\nLSTM Model Training Complete.")
        print(f"Final Out-of-Sample Test Huber Loss: {test_loss.item():.6f}")

if __name__ == "__main__":
    main()