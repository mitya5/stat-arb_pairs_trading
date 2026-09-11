import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

def hedge_ratio(y, x):
    x_ = sm.add_constant(x)
    model = sm.OLS(y, x_).fit()
    return model.params.iloc[1]

def compute_spread(y, x, beta = None):
    if not beta:
        beta = hedge_ratio(y,x)
    return y - x*beta

def half_life(spread):
    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_ret.index]
    x_ = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, x_).fit()
    theta = model.params.iloc[1]
    if theta >= 0:
        return np.inf
    return -np.log(2) / theta   

def eg_test(y, x):
    beta = hedge_ratio(y,x)
    spread = compute_spread(y,x)
    adf_stat, p_val, *_ = adfuller(spread, autolag="AIC")
    coint_stat, coint_pval, _ = coint(y, x)
    return {
        "beta": beta,
        "adf_stat": adf_stat,
        "adf_p_value": p_val,
        "coint_p_value": coint_pval,
        "is_cointegrated": p_val < 0.05,
        "half_life": half_life(spread),
    }

if __name__ == "__main__":
    # test
    np.random.seed(69)
    n = 1000
    common_trend = np.cumsum(np.random.normal(0, 1, n))
    noise_y = np.random.normal(0, 0.5, n)
    noise_x = np.random.normal(0, 0.5, n)
 
    y = pd.Series(common_trend + noise_y + 50)
    x = pd.Series(common_trend * 0.8 + noise_x + 30)
 
    result = eg_test(y, x)
    print("test:")
    for k, v in result.items():
        print(f"  {k}: {v}")