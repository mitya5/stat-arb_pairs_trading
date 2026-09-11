import os
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

def get_prices(tickers, start="2025-01-01", end=None, use_cache = True):
    import yfinance as yf
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{'_'.join(tickers)}_{start}_{end}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    data = yf.download(tickers, start, end, auto_adjust=True)
    data = data["Close"]
    data = data.dropna()
    data.to_csv(cache_path)
    return data

if __name__ == "__main__":
    # test
    df = get_prices(["KO", "PEP"])
    print(df.tail())
    print(f"\n{len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")