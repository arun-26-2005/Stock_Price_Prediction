import matplotlib
matplotlib.use('Agg')

import os
import sys

# Add parent directory to path so we can import project modules (config, models, utils)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import joblib
import yfinance as yf
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DATASETS, MODELS
from utils.data_preprocessing import preprocess_dataset
from utils.sentiment_analyzer import fetch_recent_news, analyze_sentiment, TICKER_MAP
from utils.backtest import run_backtest

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock Price Prediction API",
    description="Backend API for the Stock Price Prediction Dashboard",
    version="1.0.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_device():
    """Return best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(model_name: str, input_size: int):
    """Dynamically instantiate the correct model class based on name."""
    name = model_name.upper()
    if name == "LSTM":
        from models.lstm import LSTMModel
        return LSTMModel(input_size=input_size)
    elif name == "BILSTM":
        from models.bilstm import BiLSTMModel
        return BiLSTMModel(input_size=input_size)
    elif name == "MTRAN":
        from models.mtran import MTRAN
        return MTRAN(input_size=input_size)
    elif name == "CNN_BILSTM":
        from models.cnn_bilstm import CNN_BiLSTM
        return CNN_BiLSTM(input_size=input_size)
    elif name == "CNN_BILSTM_AM":
        from models.cnn_bilstm_am import CNN_BiLSTM_AM
        return CNN_BiLSTM_AM(input_size=input_size)
    elif name == "HYBRID":
        from models.hybrid_bilstm_mtran_tcn import BiLSTM_MTRAN_TCN
        return BiLSTM_MTRAN_TCN(input_size=input_size)
    elif name == "MTRAN_TCN":
        from models.mtran_tcn import MTRAN_TCN
        return MTRAN_TCN(input_size=input_size)
    elif name == "BILSTM_TCN":
        from models.bilstm_tcn import BiLSTM_TCN
        return BiLSTM_TCN(input_size=input_size)
    elif name == "BILSTM_MTRAN":
        from models.bilstm_mtran import BiLSTM_MTRAN
        return BiLSTM_MTRAN(input_size=input_size)
    else:
        raise ValueError(f"Unsupported model: {name}")


# Map tickers/companies to their sectors
SECTOR_MAP = {
    # IT Sector
    "TCS": "IT_SECTOR",
    "INFY": "IT_SECTOR",
    "WIPRO": "IT_SECTOR",
    "HCLTECH": "IT_SECTOR",
    "LTIM": "IT_SECTOR",
    # Energy Sector
    "RELIANCE": "ENERGY_SECTOR",
    "ONGC": "ENERGY_SECTOR",
    "NTPC": "ENERGY_SECTOR",
    "POWERGRID": "ENERGY_SECTOR",
    "COALINDIA": "ENERGY_SECTOR",
    # Banking Sector
    "HDFCBANK": "BANKING_SECTOR",
    "ICICIBANK": "BANKING_SECTOR",
    "SBIN": "BANKING_SECTOR",
    "AXISBANK": "BANKING_SECTOR",
    "KOTAKBANK": "BANKING_SECTOR",
    # Auto Sector
    "TATAMOTORS": "AUTO_SECTOR",
    "MARUTI": "AUTO_SECTOR",
    "TATASTEEL": "AUTO_SECTOR",
    "JSWSTEEL": "AUTO_SECTOR",
    # Consumer Sector
    "ITC": "CONSUMER_SECTOR",
    "HINDUNILVR": "CONSUMER_SECTOR",
    "TITAN": "CONSUMER_SECTOR",
    "NESTLEIND": "CONSUMER_SECTOR"
}

def _get_sector(stock: str) -> str:
    """Resolve the sector name for a stock, fallback to IT_SECTOR."""
    return SECTOR_MAP.get(stock.upper(), "IT_SECTOR")


def _checkpoint_path(stock: str, model_name: str = "HYBRID") -> str:
    stock_upper = stock.upper()
    # Check if a specific checkpoint exists
    specific_path = os.path.join(PROJECT_ROOT, "checkpoints", f"{stock_upper}_{model_name}.pth")
    if os.path.exists(specific_path):
        return specific_path
    
    # Fallback to Sector-grouped checkpoint
    sector = _get_sector(stock_upper)
    sector_path = os.path.join(PROJECT_ROOT, "checkpoints", f"{sector}_{model_name}.pth")
    if os.path.exists(sector_path):
        print(f"[{stock_upper}] Loading sector-fallback weights from {sector_path}")
        return sector_path
        
    # Fallback to TCS default model
    return os.path.join(PROJECT_ROOT, "checkpoints", f"TCS_{model_name}.pth")


def _scaler_path(stock: str) -> str:
    stock_upper = stock.upper()
    specific_path = os.path.join(PROJECT_ROOT, "checkpoints", f"{stock_upper}_scaler.joblib")
    if os.path.exists(specific_path):
        return specific_path
        
    # Fallback to Sector-grouped scaler
    sector = _get_sector(stock_upper)
    sector_path = os.path.join(PROJECT_ROOT, "checkpoints", f"{sector}_scaler.joblib")
    if os.path.exists(sector_path):
        print(f"[{stock_upper}] Loading sector-fallback scaler from {sector_path}")
        return sector_path
        
    # Fallback to TCS default scaler
    return os.path.join(PROJECT_ROOT, "checkpoints", "TCS_scaler.joblib")


def _has_checkpoint(stock: str) -> bool:
    # A stock is always active if sector/universal weights are present
    return True


def _create_sequences(data, window=10):
    """Create sliding-window sequences (matches train.py logic)."""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window][0])
    return np.array(X), np.array(y)


def _load_and_preprocess_csv(stock: str):
    """Load raw CSV (if local) or download dynamically in-memory from Yahoo Finance,
    then clean columns and run preprocess_dataset in-memory (no caching to disk)."""
    stock_upper = stock.upper()
    
    if stock_upper in DATASETS:
        file_path = os.path.join(PROJECT_ROOT, DATASETS[stock_upper])
        df = pd.read_csv(file_path)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df = df.drop([0, 1]).reset_index(drop=True)
        df = df.dropna().reset_index(drop=True)
        df = df.sort_values("Date")
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = preprocess_dataset(df, stock_upper, use_cache=True)
    else:
        # Download 10 years dynamically in-memory
        ticker_symbol = TICKER_MAP.get(stock_upper, f"{stock_upper}.NS")
        print(f"[/api/history] Syncing {stock_upper} ({ticker_symbol}) dynamically in-memory from Yahoo Finance...")
        start_date = "2016-01-01"
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        df_raw = yf.download(ticker_symbol, start=start_date, end=today_str, progress=False)
        
        if df_raw.empty:
            raise ValueError(f"Ticker symbol '{ticker_symbol}' not found on Yahoo Finance.")
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
            
        df = df_raw.reset_index().rename(columns={"Date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        df = df.sort_values("Date")
        # Run preprocessing indicators purely in-memory (no caching to disk)
        df = preprocess_dataset(df, stock_upper, use_cache=False)
        
    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/stocks")
async def list_stocks():
    """Return available stocks with metadata."""
    try:
        stocks = []
        for name in DATASETS:
            stocks.append({
                "name": name,
                "symbol": TICKER_MAP.get(name, name),
                "has_checkpoint": _has_checkpoint(name),
            })
        return {"stocks": stocks}
    except Exception as e:
        print(f"[/api/stocks] Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/predict/{stock}")
async def predict_stock(stock: str, model: str = "HYBRID", alpha: float = 0.015):
    """Run live prediction for a stock using a trained model + sentiment adjustment."""
    stock = stock.upper()
    model_name = model.upper()
    print(f"\n[/api/predict/{stock}] Starting prediction with model={model_name}, alpha={alpha}")

    try:
        # --- Validate stock ---
        # Allow any symbol to be requested. If not pre-configured, we download dynamically.
        symbol = TICKER_MAP.get(stock, f"{stock}.NS")

        # --- Check checkpoint ---
        ckpt_path = _checkpoint_path(stock, model_name)
        scaler_path = _scaler_path(stock)
        
        if not os.path.exists(ckpt_path) or not os.path.exists(scaler_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Base universal models not initialized. Please train TCS first."},
            )

        # --- Load or Fit Scaler ---
        if stock in ["TCS", "RELIANCE", "INFY"] and os.path.exists(scaler_path):
            print(f"[/api/predict/{stock}] Loading pre-trained scaler from {scaler_path}...")
            scaler = joblib.load(scaler_path)
        else:
            print(f"[/api/predict/{stock}] Fitting dynamic StandardScaler on historical database...")
            from sklearn.preprocessing import StandardScaler
            df_hist = _load_and_preprocess_csv(stock)
            df_hist_features = df_hist.drop(columns=["Date"])
            scaler = StandardScaler()
            scaler.fit(df_hist_features)

        # --- Download live data ---
        symbol = TICKER_MAP.get(stock, f"{stock}.NS")
        print(f"[/api/predict/{stock}] Downloading last 60 days for {symbol}...")
        df_raw = yf.download(symbol, period="60d", progress=False)
        if df_raw.empty:
            return JSONResponse(status_code=500, content={"error": "Downloaded stock data is empty"})

        df = df_raw.copy().reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna().reset_index(drop=True)

        last_trade_date = df["Date"].max().strftime("%Y-%m-%d")
        last_close_price = float(df["Close"].iloc[-1])

        # --- Preprocess ---
        print(f"[/api/predict/{stock}] Preprocessing features...")
        df_enriched = preprocess_dataset(df, stock, use_cache=False)
        df_features = df_enriched.drop(columns=["Date"])
        scaled_features = scaler.transform(df_features)

        window_size = 10
        if len(scaled_features) < window_size:
            return JSONResponse(
                status_code=500,
                content={"error": f"Not enough data ({len(scaled_features)} rows, need {window_size})"},
            )

        # --- Build input sequence ---
        last_sequence = scaled_features[-window_size:]
        X_input = np.expand_dims(last_sequence, axis=0)
        X_tensor = torch.tensor(X_input, dtype=torch.float32)

        # --- Load model ---
        input_size = X_input.shape[2]
        device = _get_device()
        mdl = _load_model(model_name, input_size)
        mdl.load_state_dict(torch.load(ckpt_path, map_location=device))
        mdl.to(device)
        mdl.eval()

        # --- Inference ---
        print(f"[/api/predict/{stock}] Running inference...")
        with torch.no_grad():
            pred_scaled = mdl(X_tensor.to(device)).cpu().numpy()

        # Inverse-scale to get actual price
        pred_array = np.zeros((1, scaled_features.shape[1]))
        pred_array[0, 0] = pred_scaled[0, 0]
        predicted_close = float(scaler.inverse_transform(pred_array)[0, 0])

        # --- Sentiment ---
        print(f"[/api/predict/{stock}] Fetching news & sentiment...")
        articles = fetch_recent_news(stock)
        sentiment_score, detailed_sentiments = analyze_sentiment(articles)

        # --- Adjust with sentiment ---
        adjustment = alpha * sentiment_score
        adjusted_close = predicted_close * (1.0 + adjustment)

        if sentiment_score >= 0.15:
            sentiment_label = "Bullish"
        elif sentiment_score <= -0.15:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        change_pct = ((adjusted_close - last_close_price) / last_close_price) * 100

        print(f"[/api/predict/{stock}] Done. Adjusted price={adjusted_close:.2f}")

        return {
            "stock": stock,
            "symbol": symbol,
            "last_date": last_trade_date,
            "last_close": round(last_close_price, 2),
            "baseline_price": round(predicted_close, 2),
            "adjusted_price": round(adjusted_close, 2),
            "change_pct": round(change_pct, 2),
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_label": sentiment_label,
            "articles": detailed_sentiments[:10],  # cap at 10
        }

    except Exception as e:
        print(f"[/api/predict/{stock}] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/history/{stock}")
async def stock_history(stock: str):
    """Return processed historical data with technical indicators for charting."""
    stock = stock.upper()
    print(f"\n[/api/history/{stock}] Loading historical data...")

    try:
        # Allow any symbol to be requested. If not pre-configured, we download dynamically.

        df = _load_and_preprocess_csv(stock)

        # Convert Date column to string for JSON serialization
        result = {}
        for col in df.columns:
            if col == "Date":
                result["dates"] = df["Date"].astype(str).tolist()
            else:
                # Convert to python floats, replace NaN with None
                vals = df[col].tolist()
                result[col.lower()] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in vals]

        print(f"[/api/history/{stock}] Returning {len(df)} rows")
        return result

    except Exception as e:
        print(f"[/api/history/{stock}] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/backtest/{stock}")
async def backtest_stock(stock: str, model: str = "HYBRID"):
    """Run backtest on trained model predictions."""
    stock = stock.upper()
    model_name = model.upper()
    print(f"\n[/api/backtest/{stock}] Starting backtest with model={model_name}")

    try:
        # --- Check checkpoint ---
        ckpt_path = _checkpoint_path(stock, model_name)
        scaler_path = _scaler_path(stock)
        
        if not os.path.exists(ckpt_path) or not os.path.exists(scaler_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Base universal models not initialized. Please train TCS first."},
            )

        # --- Load and preprocess data (same flow as train.py) ---
        print(f"[/api/backtest/{stock}] Loading and preprocessing dataset...")
        df = _load_and_preprocess_csv(stock)

        # --- Load or Fit Scaler ---
        if stock in ["TCS", "RELIANCE", "INFY"] and os.path.exists(scaler_path):
            print(f"[/api/backtest/{stock}] Loading pre-trained scaler from {scaler_path}...")
            scaler = joblib.load(scaler_path)
        else:
            print(f"[/api/backtest/{stock}] Fitting dynamic StandardScaler on historical database...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(df.drop(columns=["Date"]))
        df_features = df.drop(columns=["Date"])
        scaled_data = scaler.transform(df_features)
        scaled_data = np.nan_to_num(scaled_data)

        # --- Create sequences ---
        window_size = 10
        X_all, y_all = _create_sequences(scaled_data, window_size)

        # --- Split 80/20 for test set ---
        split_idx = int(len(X_all) * 0.8)
        X_test = X_all[split_idx:]
        y_test = y_all[split_idx:]

        if len(X_test) == 0:
            return JSONResponse(
                status_code=500,
                content={"error": "Not enough data for backtesting"},
            )

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # --- Load model ---
        print(f"[/api/backtest/{stock}] Loading model...")
        input_size = X_test.shape[2]
        device = _get_device()
        mdl = _load_model(model_name, input_size)
        mdl.load_state_dict(torch.load(ckpt_path, map_location=device))
        mdl.to(device)
        mdl.eval()

        # --- Run predictions on test set ---
        print(f"[/api/backtest/{stock}] Running inference on test set ({len(X_test)} samples)...")
        predictions = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i : i + batch_size].to(device)
                preds = mdl(batch)
                predictions.extend(preds.squeeze().cpu().numpy().tolist())

        predictions = np.array(predictions)

        # --- Inverse-scale predictions and actuals ---
        num_features = scaled_data.shape[1]

        pred_array = np.zeros((len(predictions), num_features))
        pred_array[:, 0] = predictions
        pred_original = scaler.inverse_transform(pred_array)[:, 0]

        actual_array = np.zeros((len(y_test), num_features))
        actual_array[:, 0] = y_test
        actual_original = scaler.inverse_transform(actual_array)[:, 0]

        # --- Get corresponding dates ---
        # After sequences with window_size=10, index offset = window_size
        # Test set starts at split_idx, so date indices = split_idx + window_size ... onwards
        date_start = split_idx + window_size
        test_dates = df["Date"].iloc[date_start : date_start + len(y_test)]

        # --- Run backtest ---
        print(f"[/api/backtest/{stock}] Running backtest strategy...")
        backtest_results = run_backtest(
            actual_prices=actual_original,
            predicted_prices=pred_original,
            dates=test_dates,
            initial_capital=100000.0,
            transaction_cost=0.001,
            threshold=0.0025,
        )

        print(f"[/api/backtest/{stock}] Done. Return={backtest_results['total_return']:.2f}%")

        return {
            "portfolio_values": [float(v) for v in backtest_results["portfolio_values"]],
            "benchmark_values": [float(v) for v in backtest_results["benchmark_values"]],
            "dates": [str(d) for d in backtest_results["dates"]],
            "total_return": round(float(backtest_results["total_return"]), 2),
            "bench_return": round(float(backtest_results["bench_return"]), 2),
            "sharpe_ratio": round(float(backtest_results["sharpe_ratio"]), 2),
            "bench_sharpe": round(float(backtest_results["bench_sharpe"]), 2),
            "max_drawdown": round(float(backtest_results["max_drawdown"]), 2),
            "bench_max_drawdown": round(float(backtest_results["bench_max_drawdown"]), 2),
            "num_trades": int(backtest_results["num_trades"]),
            "win_rate": round(float(backtest_results["win_rate"]), 2),
        }

    except Exception as e:
        print(f"[/api/backtest/{stock}] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/sentiment/{stock}")
async def stock_sentiment(stock: str):
    """Fetch and analyze current news sentiment for a stock."""
    stock = stock.upper()
    print(f"\n[/api/sentiment/{stock}] Fetching news sentiment...")

    try:
        if stock not in DATASETS:
            return JSONResponse(status_code=404, content={"error": f"Unknown stock: {stock}"})

        articles = fetch_recent_news(stock)
        sentiment_score, detailed_sentiments = analyze_sentiment(articles)

        if sentiment_score >= 0.15:
            label = "Bullish"
        elif sentiment_score <= -0.15:
            label = "Bearish"
        else:
            label = "Neutral"

        print(f"[/api/sentiment/{stock}] Score={sentiment_score:.4f}, Label={label}")

        return {
            "score": round(sentiment_score, 4),
            "label": label,
            "articles": detailed_sentiments,
        }

    except Exception as e:
        print(f"[/api/sentiment/{stock}] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/macro/{stock}")
async def macro_correlation(stock: str):
    """Compute Pearson correlation matrix between stock close and macro features."""
    stock = stock.upper()
    print(f"\n[/api/macro/{stock}] Computing macro correlation matrix...")

    try:
        # Allow any symbol to be requested. We compute correlation dynamically in-memory.

        df = _load_and_preprocess_csv(stock)

        # Select the columns we want for correlation
        target_cols = [
            "Close",
            "Gold_Close",
            "Crude_Close",
            "USD_INR_Close",
            "SP500_Close",
            "VIX_Close",
            "Nifty50_Close",
        ]
        available_cols = [c for c in target_cols if c in df.columns]
        df_subset = df[available_cols].apply(pd.to_numeric, errors="coerce")

        corr_matrix = df_subset.corr()

        # Replace NaN with None for JSON serialization
        matrix_values = corr_matrix.values.tolist()
        clean_matrix = []
        for row in matrix_values:
            clean_row = [None if (isinstance(v, float) and np.isnan(v)) else round(v, 4) for v in row]
            clean_matrix.append(clean_row)

        print(f"[/api/macro/{stock}] Correlation matrix computed ({len(available_cols)} features)")

        return {
            "features": available_cols,
            "matrix": clean_matrix,
        }

    except Exception as e:
        print(f"[/api/macro/{stock}] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/models")
async def list_models():
    """List available model architectures from config."""
    try:
        models = []
        for name, module_name in MODELS.items():
            models.append({
                "name": name,
                "module_name": module_name,
            })
        return {"models": models}
    except Exception as e:
        print(f"[/api/models] Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# In-memory dictionary to track background retraining task status
retrain_tasks = {}

def run_retrain_task(stock: str, epochs: int):
    try:
        retrain_tasks[stock] = {"status": "running", "message": "Synchronizing Yahoo Finance and fine-tuning checkpoints..."}
        from retrain import retrain_model
        total_rows = retrain_model(stock, epochs=epochs)
        retrain_tasks[stock] = {
            "status": "success", 
            "message": f"Successfully updated sliding window dataset and fine-tuned checkpoints! Total window rows: {total_rows}"
        }
    except Exception as e:
        print(f"Retrain background task error for {stock}: {e}")
        retrain_tasks[stock] = {"status": "failed", "message": str(e)}

@app.post("/api/retrain/{stock}")
async def trigger_retrain(stock: str, background_tasks: BackgroundTasks, epochs: int = 15):
    stock_upper = stock.upper()
    if stock_upper not in DATASETS:
        return JSONResponse(status_code=400, content={"error": f"Invalid stock name: {stock}"})
    
    # Check if a task is already running for this stock
    if retrain_tasks.get(stock_upper, {}).get("status") == "running":
        return {"status": "running", "message": "Retraining is already in progress for this asset."}
        
    background_tasks.add_task(run_retrain_task, stock_upper, epochs)
    return {"status": "started", "message": "Continuous retraining task successfully started in the background."}

@app.get("/api/retrain/status/{stock}")
async def get_retrain_status(stock: str):
    stock_upper = stock.upper()
    status_info = retrain_tasks.get(stock_upper, {"status": "idle", "message": "Ready to sync."})
    return status_info

