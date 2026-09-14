import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import get_prices
from cointegration import eg_test
from backtest import run_backtest, performance_metrics


TICKERS = [["JPM", "BAC"], ["KO", "PEP"], ["GM", "F"], ["AAPL", "MSFT"], ["V", "MA"]]
START = "2017-01-01"

def main():
    lowest_p = np.inf
    best_pair = []
    for list_t in TICKERS:
        prices = get_prices(list_t, start = START)
        y, x = prices[list_t[0]], prices[list_t[1]]
        p_val = eg_test(y,x)["adf_p_value"]
        if p_val < lowest_p:
            best_pair = list_t
            lowest_p = p_val
    print(f"best pair was {best_pair}")
    print("full sample EG test, for reference")
    prices = get_prices(best_pair, start = START)
    y, x = prices[best_pair[0]], prices[best_pair[1]]
    coint_result = eg_test(y, x)
    for k, v in coint_result.items():
            print(f"{k}: {v}")
    if not coint_result["is_cointegrated"]:
        print("best pair did not pass test")
        return    
    results = run_backtest(y, x, window=30, entry=2.0, exit=0.5, cost=5.0)
    metrics = performance_metrics(results)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    axes[0].plot(prices.index, y, label=best_pair[0])
    axes[0].plot(prices.index, x, label=best_pair[1])
    axes[0].set_title(f"{best_pair[0]} vs {best_pair[1]}")
    axes[0].legend()
    axes[1].plot(results.index, results["zscore"], color="purple")
    axes[1].axhline(2.0, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(-2.0, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("z_score with thresholds")
    axes[2].plot(results.index, results["equity_curve"], color="green")
    axes[2].set_title("equity curve")
    plt.tight_layout()
    out_path = f"{best_pair[0]}_{best_pair[1]}_{START}_current_backtest_report.png"
    plt.savefig(out_path, dpi=120)

if __name__ == "__main__":
    main()


    

