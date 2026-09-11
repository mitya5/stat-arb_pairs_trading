import numpy as np
import pandas as pd
from cointegration import hedge_ratio

def walk_forward_beta(y, x, min_window = 252):
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(min_window, len(y)):
        betas.iloc[i] = hedge_ratio(y.iloc[:i], x.iloc[:i])
    return betas.ffill()

def run_backtest(y, x, window=30, entry = 2.0, exit = 0.5, min_window = 252, cost = 5.0):
    from strategy import gen_signal
    beta = walk_forward_beta(y,x,min_window)
    spread = y - beta*x
    sig = gen_signal(spread.dropna(), window=window, entry=entry, exit=exit)
    y_aligned = y.reindex(sig.index)
    x_aligned = x.reindex(sig.index)
    beta_aligned = beta.reindex(sig.index)
    notional = y_aligned.abs() + (beta_aligned.abs() * x_aligned.abs())
    spread_change = sig["spread"].diff()
    gross_return = sig["pos"].shift(1) * spread_change / notional
    position_change = sig["pos"].diff().abs().fillna(0)
    cost = position_change * (cost / 10_000) * 2
    net_return = gross_return.fillna(0) - cost
    out = sig.copy()
    out["beta"] = beta.reindex(sig.index)
    out["gross_return"] = gross_return
    out["cost"] = cost
    out["net_return"] = net_return
    out["equity_curve"] = (1 + net_return.fillna(0)).cumprod()
    return out

def performance_metrics(results, p_per_year = 252):
    r = results["net_return"].dropna()
    equity = results["equity_curve"].dropna()
    if r.std() > 0:
        sharpe = r.mean() / r.std() * np.sqrt(p_per_year)
    else:
        sharpe = np.nan
    running_max = equity.cummax()
    cur_drawdown = (equity - running_max)/running_max
    max_drawdown = cur_drawdown.min()
    total_return = equity.iloc[-1] - 1
    n_trades = (results["pos"].diff().abs() > 0).sum()
    return {"total return": total_return, "annualized_sharpe": sharpe,
            "max_drawdown": max_drawdown, "trades": int(n_trades)}

if __name__ == "__main__":
    # test
    np.random.seed(1)
    n = 1500
    common = np.cumsum(np.random.normal(0, 1, n))
    y = pd.Series(common + np.random.normal(0, 0.5, n) + 100)
    x = pd.Series(common * 0.9 + np.random.normal(0, 0.5, n) + 60)
 
    results = run_backtest(y, x)
    metrics = performance_metrics(results)
    print("test:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    
    