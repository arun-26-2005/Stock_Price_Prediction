import os
import sys
import argparse
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Ensure we can import from the parent directory / config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASETS, MODELS
from utils.data_preprocessing import preprocess_dataset
from utils.sentiment_analyzer import TICKER_MAP

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])  # Close price is column 0
    return np.array(X), np.array(y)

def update_raw_csv_sliding_window(stock, ticker_symbol, csv_path, max_rows=2600):
    """
    Downloads new daily data from the last date in the CSV up to today,
    appends it, keeps the last max_rows rows (sliding window),
    and saves it back in the original custom header format.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source raw CSV not found at {csv_path}")

    # Read existing custom format CSV
    raw_df = pd.read_csv(csv_path)
    clean_df = raw_df.copy()
    clean_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    clean_df = clean_df.drop([0, 1]).reset_index(drop=True)
    clean_df["Date"] = pd.to_datetime(clean_df["Date"])
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    
    # Get last date in CSV
    last_date = clean_df["Date"].max()
    print(f"[{stock}] Last date in local CSV: {last_date.strftime('%Y-%m-%d')}")
    
    # Download new candles starting from day after last_date to today
    start_download = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    if pd.to_datetime(start_download) < pd.to_datetime(today_str):
        print(f"[{stock}] Syncing dataset. Downloading new data from {start_download} to {today_str}...")
        new_data = yf.download(ticker_symbol, start=start_download, end=today_str, progress=False)
        
        if not new_data.empty:
            # Clean yfinance multi-index columns if present
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)
                
            new_df = pd.DataFrame()
            new_df["Date"] = new_data.index
            new_df["Close"] = new_data["Close"].values
            new_df["High"] = new_data["High"].values
            new_df["Low"] = new_data["Low"].values
            new_df["Open"] = new_data["Open"].values
            new_df["Volume"] = new_data["Volume"].values
            
            # Combine raw data
            appended_df = pd.concat([clean_df, new_df], ignore_index=True)
            # Remove any duplicates on Date
            appended_df = appended_df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            print(f"[{stock}] Appended {len(new_df)} new rows from Yahoo Finance.")
        else:
            print(f"[{stock}] No new market data returned from Yahoo Finance.")
            appended_df = clean_df
    else:
        print(f"[{stock}] Local CSV dataset is already up-to-date.")
        appended_df = clean_df
        
    # Enforce sliding window (keep last max_rows rows, dropping oldest month of data if it overflows)
    if len(appended_df) > max_rows:
        rows_dropped = len(appended_df) - max_rows
        print(f"[{stock}] Sliding window trigger: Dropping oldest {rows_dropped} rows to maintain 10-year limit ({max_rows} rows).")
        appended_df = appended_df.tail(max_rows).reset_index(drop=True)
        
    # Write back to raw CSV path in the original format
    with open(csv_path, 'w') as f:
        f.write("Price,Close,High,Low,Open,Volume\n")
        f.write(f"Ticker,{ticker_symbol},{ticker_symbol},{ticker_symbol},{ticker_symbol},{ticker_symbol}\n")
        f.write("Date,,,,,\n")
        for _, row in appended_df.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            f.write(f"{date_str},{row['Close']},{row['High']},{row['Low']},{row['Open']},{row['Volume']}\n")
            
    print(f"[{stock}] Raw CSV successfully written. Total rows: {len(appended_df)}")
    return appended_df

def retrain_model(stock, epochs=10, lr=0.00005):
    """
    Pipeline to sync raw data, run sliding window, compute technical indicators,
    and fine-tune the saved deep learning checkpoint with the fresh price data.
    """
    stock = stock.upper()
    if stock not in DATASETS:
        raise ValueError(f"Stock must be one of {list(DATASETS.keys())}")
        
    ticker_symbol = TICKER_MAP[stock]
    csv_path = DATASETS[stock]
    
    # 1. Sync raw data and slide window
    df_raw = update_raw_csv_sliding_window(stock, ticker_symbol, csv_path)
    
    # 2. Re-run preprocessing (force re-calc technical indicators & download new macro features)
    df_processed = preprocess_dataset(df_raw, stock, use_cache=False)
    
    # 3. Create splits
    train_df = df_processed[df_processed['Date'] < '2023-01-01']
    test_df = df_processed[df_processed['Date'] >= '2023-01-01']
    
    # 4. Scale features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.drop(columns=['Date']))
    test_scaled = scaler.transform(test_df.drop(columns=['Date']))
    
    # Save the updated scaler
    scaler_path = os.path.join("checkpoints", f"{stock}_scaler.joblib")
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[{stock}] Saved updated scaler to {scaler_path}")
    
    # 5. Create sequence tensors
    X_train, y_train = _create_sequences(train_scaled, window_size=10)
    X_test, y_test = _create_sequences(test_scaled, window_size=10)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # 6. Initialize model
    from models.hybrid_bilstm_mtran_tcn import BiLSTM_MTRAN_TCN
    model = BiLSTM_MTRAN_TCN(input_size=X_train.shape[2])
    
    # Load existing checkpoint if it exists
    checkpoint_path = os.path.join("checkpoints", f"{stock}_HYBRID.pth")
    if os.path.exists(checkpoint_path):
        print(f"[{stock}] Loading existing checkpoint for fine-tuning: {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"[{stock}] Checkpoint not found. Initializing a new model training session.")
        
    # 7. Fine-tuning Loop
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[{stock}] Retraining model on device: {device}...")
    model.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{stock}] Epoch {epoch+1}/{epochs} | Loss: {total_loss:.5f}")
        
    # Save the updated model weights
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[{stock}] Successfully saved updated model weights to {checkpoint_path}")
    return len(df_raw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding-window incremental retraining pipeline.")
    parser.add_argument("--stock", type=str, required=True, help="Stock name (e.g. TCS)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    args = parser.parse_args()
    
    try:
        retrain_model(args.stock, epochs=args.epochs)
    except Exception as e:
        print(f"Error during retraining pipeline: {e}")
        sys.exit(1)
