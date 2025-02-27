import numpy as np

def calculate_z_scores(df, window_short=20, window_medium=50, window_long=100):
    df["spread_ma_short"] = df["spread"].rolling(window=window_short).mean()
    df["spread_std_short"] = df["spread"].rolling(window=window_short).std()
    df["z_score_short"] = (df["spread"] - df["spread_ma_short"]) / df["spread_std_short"].replace(0, np.nan)
    
    df["spread_ma_med"] = df["spread"].rolling(window=window_medium).mean()
    df["spread_std_med"] = df["spread"].rolling(window=window_medium).std()
    df["z_score"] = df["z_score_med"] = (df["spread"] - df["spread_ma_med"]) / df["spread_std_med"].replace(0, np.nan)
    
    df["spread_ma_long"] = df["spread"].rolling(window=window_long).mean()
    df["spread_std_long"] = df["spread"].rolling(window=window_long).std()
    df["z_score_long"] = (df["spread"] - df["spread_ma_long"]) / df["spread_std_long"].replace(0, np.nan)
    
    return df

def calculate_trend(xauusd):
    # Calcula EMAs para XAUUSD e define a tendência
    ema20 = xauusd["close"].ewm(span=20, adjust=False).mean()
    ema50 = xauusd["close"].ewm(span=50, adjust=False).mean()
    trend = np.where(ema20 > ema50, 1, -1)
    trend_series = pd.Series(trend, index=xauusd.index)
    return trend_series
