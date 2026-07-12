import os
import argparse
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
from config import DATASETS, MODELS
from utils.data_preprocessing import preprocess_dataset
from utils.sentiment_analyzer import fetch_recent_news, analyze_sentiment, TICKER_MAP

def main():
    parser = argparse.ArgumentParser(description="Live Stock Prediction with News Sentiment Override")
    parser.add_argument("--dataset", type=str, default="TCS", help="Stock dataset/ticker to predict (TCS, RELIANCE, INFY, NIFTY)")
    parser.add_argument("--model", type=str, default="HYBRID", help="Model architecture (e.g. HYBRID, LSTM, BILSTM)")
    parser.add_argument("--alpha", type=float, default=0.015, help="Maximum percentage impact of news sentiment (e.g. 0.015 = 1.5%)")
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    model_name = args.model.upper()

    print("=" * 66)
    print(f"STARTING LIVE STOCK PREDICTION FOR {dataset_name}")
    print("=" * 66)

    # 1. Load trained model checkpoint and scaler
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_{model_name}.pth")
    scaler_path = os.path.join(checkpoint_dir, f"{dataset_name}_scaler.joblib")

    if not os.path.exists(checkpoint_path) or not os.path.exists(scaler_path):
        print(f"Error: Model checkpoint or scaler not found!")
        print(f"Please train the model first by running:")
        print(f"  python train.py --dataset {dataset_name} --model {model_name}")
        return

    print("Loading scaler and model checkpoint...")
    scaler = joblib.load(scaler_path)

    # 2. Download recent stock data for sequence creation
    symbol = TICKER_MAP.get(dataset_name, dataset_name)
    print(f"Downloading last 60 days of historical stock data for {symbol}...")
    try:
        df_raw = yf.download(symbol, period="60d", progress=False)
        if df_raw.empty:
            print("Error: Downloaded stock data is empty.")
            return
            
        df = df_raw.copy()
        df = df.reset_index()
        
        # Clean multi-index columns if present (can happen in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        # Reformat columns to match raw CSV schema
        df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna().reset_index(drop=True)
        
        last_trade_date = df['Date'].max().strftime('%Y-%m-%d')
        last_close_price = float(df['Close'].iloc[-1])
        
    except Exception as e:
        print(f"Error retrieving live stock data: {e}")
        return

    # 3. Preprocess and enrich features dynamically
    print("Calculating technical indicators and downloading macroeconomic features...")
    try:
        # We set use_cache=False to prevent loading outdated historical cached CSVs
        df_enriched = preprocess_dataset(df, dataset_name, use_cache=False)
    except Exception as e:
        print(f"Error during feature enrichment: {e}")
        return

    # 4. Construct sequence input
    df_features = df_enriched.drop(columns=['Date'])
    
    # Scale features using the loaded scaler
    try:
        scaled_features = scaler.transform(df_features)
    except Exception as e:
        print(f"Error scaling features. Ensure model structure matches saved scaler: {e}")
        return

    window_size = 10
    if len(scaled_features) < window_size:
        print(f"Error: Preprocessed dataset has only {len(scaled_features)} rows. Needs at least {window_size} rows.")
        return

    # Extract the most recent sequence (last 10 trading days)
    last_sequence = scaled_features[-window_size:]
    X_input = np.expand_dims(last_sequence, axis=0) # shape [1, 10, num_features]
    X_tensor = torch.tensor(X_input, dtype=torch.float32)

    # 5. Instantiate model and load state
    input_size = X_input.shape[2]
    
    if model_name == "LSTM":
        from models.lstm import LSTMModel
        model = LSTMModel(input_size=input_size)
    elif model_name == "BILSTM":
        from models.bilstm import BiLSTMModel
        model = BiLSTMModel(input_size=input_size)
    elif model_name == "MTRAN":
        from models.mtran import MTRAN
        model = MTRAN(input_size=input_size)
    elif model_name == "CNN_BILSTM":
        from models.cnn_bilstm import CNN_BiLSTM
        model = CNN_BiLSTM(input_size=input_size)
    elif model_name == "CNN_BILSTM_AM":
        from models.cnn_bilstm_am import CNN_BiLSTM_AM
        model = CNN_BiLSTM_AM(input_size=input_size)
    elif model_name == "HYBRID":
        from models.hybrid_bilstm_mtran_tcn import BiLSTM_MTRAN_TCN
        model = BiLSTM_MTRAN_TCN(input_size=input_size)
    else:
        print(f"Unsupported model: {model_name}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Generate baseline forecast
    print("Generating baseline time-series trend prediction...")
    with torch.no_grad():
        pred_scaled = model(X_tensor.to(device)).cpu().numpy()

    # Inverse scale prediction
    pred_array = np.zeros((1, scaled_features.shape[1]))
    pred_array[0, 0] = pred_scaled[0, 0]
    predicted_close = float(scaler.inverse_transform(pred_array)[0, 0])

    # 6. Fetch news and calculate sentiment
    print("Fetching today's live stock news and analyzing sentiment...")
    articles = fetch_recent_news(dataset_name)
    sentiment_score, detailed_sentiments = analyze_sentiment(articles)

    # 7. Apply sentiment adjustment
    # Price_adj = Price_base * (1 + alpha * sentiment_score)
    adjustment = args.alpha * sentiment_score
    adjusted_close = predicted_close * (1.0 + adjustment)

    # Determine sentiment label and category
    if sentiment_score >= 0.15:
        sentiment_label = "Bullish"
        sentiment_color = "\033[92m" # Green
    elif sentiment_score <= -0.15:
        sentiment_label = "Bearish"
        sentiment_color = "\033[91m" # Red
    else:
        sentiment_label = "Neutral"
        sentiment_color = "\033[94m" # Blue

    change_pct = ((adjusted_close - last_close_price) / last_close_price) * 100
    change_color = "\033[92m" if change_pct >= 0 else "\033[91m"
    reset_color = "\033[0m"

    # 8. Render dashboard
    print("\n" + "=" * 66)
    print("                   LIVE STOCK PREDICTION REPORT                   ")
    print("=" * 66)
    print(f" Stock Symbol            : {symbol}")
    print(f" Last Traded Date        : {last_trade_date}")
    print(f" Last Closing Price      : INR {last_close_price:,.2f}")
    print("-" * 66)
    print(" --- News Sentiment Overview ---")
    print(f" Articles Analyzed       : {len(articles)}")
    print(f" Consolidated Sentiment  : {sentiment_color}{sentiment_score:+.2f} ({sentiment_label}){reset_color}")
    print("-" * 66)
    print(" --- Headline Sentiments ---")
    if not detailed_sentiments:
        print("  No recent news articles available to analyze.")
    else:
        for idx, item in enumerate(detailed_sentiments[:5]):
            color = "\033[92m" if item['sentiment'] == 'positive' else ("\033[91m" if item['sentiment'] == 'negative' else "\033[94m")
            print(f"  {idx+1}. [{color}{item['sentiment'].upper()}{reset_color}] {item['title'][:55]}...")
        if len(detailed_sentiments) > 5:
            print(f"  ... and {len(detailed_sentiments)-5} more articles.")
    print("-" * 66)
    print(" --- Tomorrow's Price Forecast ---")
    print(f" Baseline Trend Price    : INR {predicted_close:,.2f}")
    print(f" News-Adjusted Price     : INR {adjusted_close:,.2f}")
    print(f" Predicted Change (vs Today): {change_color}{change_pct:+.2f}%{reset_color}")
    print("=" * 66 + "\n")

if __name__ == "__main__":
    main()
