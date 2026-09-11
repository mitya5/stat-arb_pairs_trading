import pandas as pd

def rolling_z(spread, window):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread-mean)/std

def gen_signal(spread, window=30, entry=2.0, exit=.5):
    z = rolling_z(spread, window)
    pos = pd.Series(index = spread.index, dtype = float)
    pos.iloc[:] = float('nan')
    pos[z > entry] = -1.0
    pos[z < -entry ] = 1
    pos[z.abs() < exit] = 0.0
    pos = pos.ffill().fillna(0.0)
    return pd.DataFrame({"spread": spread, "zscore": z, "pos": pos})
