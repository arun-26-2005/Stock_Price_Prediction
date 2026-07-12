import os
import pandas as pd
import yfinance as yf
import numpy as np

def download_macro_feature(ticker, start_date, end_date, col_name):
    """
    Downloads historical close price for a macro ticker, cleans column indices,
    and returns a clean DataFrame with columns ['Date', col_name].
    """
    print(f"Downloading macro ticker: {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"Warning: downloaded data for {ticker} is empty.")
            return None
            
        # Clean multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                close_series = data['Close'].iloc[:, 0]
            else:
                close_series = data.iloc[:, 0]
        else:
            if 'Close' in data.columns:
                close_series = data['Close']
            else:
                close_series = data.iloc[:, 0]
                
        # Format index
        df_macro = pd.DataFrame(close_series).rename(columns={close_series.name: col_name})
        df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None)
        df_macro.index.name = 'Date'
        df_macro = df_macro.reset_index()
        return df_macro
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """
    Computes professional technical indicators: MA5, MA10, MA20, EMA12, EMA26,
    MACD, MACD_Signal, MACD_Hist, RSI, BB_High, BB_Low, ATR.
    """
    df = df.copy()
    
    # 1. Moving Averages & EMAs
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 2. MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 3. RSI (14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands (20)
    std20 = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['MA20'] + (std20 * 2)
    df['BB_Low'] = df['MA20'] - (std20 * 2)
    
    # 5. ATR (14)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    
    return df

def preprocess_dataset(df_raw, ticker_name, cache_dir="data", use_cache=True):
    """
    Cleans stock data, downloads and merges macro features, computes technical indicators,
    caches the result, and returns the fully enriched DataFrame.
    """
    cache_path = os.path.join(cache_dir, f"processed_{ticker_name}.csv")
    
    if use_cache and os.path.exists(cache_path):
        print(f"Loading preprocessed data for {ticker_name} from cache: {cache_path}")
        df_processed = pd.read_csv(cache_path)
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        return df_processed
        
    print(f"Preprocessing raw data and generating features for {ticker_name}...")
    
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define date range for macro indicators
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = (df['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Macro tickers dictionary
    # Gold (GC=F), Brent Crude (BZ=F), USD/INR (INR=X), S&P 500 (^GSPC), Volatility Index (^VIX)
    macro_tickers = {
        'GC=F': 'Gold_Close',
        'BZ=F': 'Crude_Close',
        'INR=X': 'USD_INR_Close',
        '^GSPC': 'SP500_Close',
        '^VIX': 'VIX_Close'
    }
    
    # Add domestic benchmark (NIFTY 50) if target ticker is not NIFTY itself
    if ticker_name != 'NIFTY':
        macro_tickers['^NSEI'] = 'Nifty50_Close'
        
    for ticker, col_name in macro_tickers.items():
        df_macro = download_macro_feature(ticker, start_date, end_date, col_name)
        if df_macro is not None:
            df = pd.merge(df, df_macro, on='Date', how='left')
            
    # Forward-fill / Backward-fill missing macro data (e.g. holiday mismatches)
    # Exclude 'Date' column from ffill/bfill to prevent date corruption
    date_col = df['Date']
    cols_to_fill = df.columns.drop('Date')
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()
    
    # Compute technical indicators
    df = add_technical_indicators(df)
    
    # Drop rows with NaNs caused by indicators (e.g. first 20 rows for moving averages)
    df = df.dropna().reset_index(drop=True)
    
    # Cache processed dataset if enabled
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved processed dataset for {ticker_name} to cache: {cache_path}")
        
    return df
