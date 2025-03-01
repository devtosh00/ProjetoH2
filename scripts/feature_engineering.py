import pandas as pd
import numpy as np
import logging
import talib

def add_indicators(df, cooldown_period=24):
    """Adiciona indicadores técnicos e features avançadas ao DataFrame."""

    logging.info("📊 Adicionando indicadores técnicos e features avançadas...")

    if df is None or df.empty:
        logging.error("❌ DataFrame vazio ou None passado para add_indicators")
        return df

    if cooldown_period <= 0:
        cooldown_period = 24  # Valor padrão
        logging.warning(f"⚠️ cooldown_period inválido, usando valor padrão: {cooldown_period}")

    df = df.copy()

    # 🔹 **Médias móveis do spread**  
    window_sizes = [24, 14 * 7, 14 * 24, 30 * 24, 60 * 24]  # Convertendo dias para horas
    for window in window_sizes:
        df[f'spread_ma_{window}'] = df['spread'].rolling(window=window, min_periods=1).mean()

    # 🔹 **Cálculo de Indicadores Separados para PAXG (X) e XAUUSD (Y)**

    for suffix in ["_x", "_y"]:  # Para cada ativo (PAXG e XAUUSD)
        close_col = f"close{suffix}"
        high_col = f"high{suffix}"
        low_col = f"low{suffix}"

        # MACD - Moving Average Convergence Divergence
        df[f"macd{suffix}"], df[f"macd_signal{suffix}"], df[f"macd_hist{suffix}"] = talib.MACD(
            df[close_col], fastperiod=12 * 24, slowperiod=26 * 24, signalperiod=9 * 24
        )

        # Stochastic Oscillator
        df[f"stoch_k{suffix}"], df[f"stoch_d{suffix}"] = talib.STOCH(
            df[high_col], df[low_col], df[close_col], fastk_period=14 * 24, slowk_period=3 * 24, slowd_period=3 * 24
        )

        # RSI - Relative Strength Index
        df[f"rsi{suffix}"] = talib.RSI(df[close_col], timeperiod=14 * 24)

        # ADX - Average Directional Index
        df[f"adx{suffix}"] = talib.ADX(df[high_col], df[low_col], df[close_col], timeperiod=14 * 24)

        # ATR - Average True Range
        df[f"atr{suffix}"] = talib.ATR(df[high_col], df[low_col], df[close_col], timeperiod=48)

        # Médias móveis para cada ativo
        for window in [24, 7 * 24, 14 * 24, 30 * 24, 60 * 24]:
            df[f"ma_{window}{suffix}"] = df[close_col].rolling(window=window, min_periods=1).mean()

    # 🔹 **Bollinger Bands para o Spread**
    df["upper_band"], df["middle_band"], df["lower_band"] = talib.BBANDS(df["spread"], timeperiod=20 * 24)

    # 🔹 **Volatilidade do spread (desvio padrão)**
    df['spread_std'] = df['spread'].rolling(window=30 * 24, min_periods=1).std()

    # 🔹 **Médias móveis da volatilidade do spread**
    df['spread_std_ma'] = df['spread_std'].rolling(window=14 * 24, min_periods=1).mean()

    # 🔹 **Tendência do spread**
    df['trend'] = np.sign(df['spread'].diff(14 * 24))  # Diferencial em 14 dias
    df['trend_ma'] = df['trend'].rolling(window=14 * 24, min_periods=1).mean()

    # 🔹 **Feature de Cooldown**
    df["cooldown"] = 0
    df.loc[df.index.factorize()[0] % cooldown_period == 0, "cooldown"] = 1

    # 🔹 **Cadeia de Markov (Probabilidade de Persistência do Estado)**
    df = add_markov_features(df, column="spread", n_states=3, window=100)

    # Preencher valores NaN com a mediana da coluna correspondente
    nan_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    for col in nan_cols:
        df[col].fillna(df[col].median(), inplace=True)

    if nan_cols:
        logging.warning(f"⚠️ Ainda existem valores NaN nas colunas: {nan_cols}")

    # Ordenar as colunas do DataFrame
    df = df[sorted(df.columns)]

    logging.info(f"✅ Indicadores adicionados com sucesso. Dimensões do DataFrame: {df.shape}")
    return df


def add_markov_features(df, column="spread", n_states=3, window=100):
    
    """Adiciona probabilidade de transição da Cadeia de Markov ao DataFrame."""

    df = df.copy()
    df["state"] = pd.qcut(df[column], q=n_states, labels=False, duplicates="drop")

    markov_prob = np.full(len(df), np.nan)
    for i in range(window, len(df)):
        window_states = df["state"].iloc[i-window:i].values
        if len(window_states) < 2:
            continue
        same = np.sum(window_states[1:] == window_states[:-1])
        markov_prob[i] = same / (len(window_states) - 1)

    df["markov_prob_same"] = markov_prob
    df["markov_prob_same"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["markov_prob_same"].fillna(df["markov_prob_same"].median(), inplace=True)

    df.drop(columns=["state"], inplace=True)
    logging.info("✅ Features de cadeia de Markov adicionadas com sucesso!")

    return df

def create_gold_target(df):
    """
    Cria target para COMPRA e VENDA baseado em padrões temporais.
    """
    # Fazer uma cópia completa para evitar problemas de view x copy
    df = df.copy()
    
    # Adicionar features temporais ao DataFrame principal
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    
    # Dividir os dados em treino e teste (treino até 2022)
    train_mask = df.index < '2023-01-01'
    train_data = df[train_mask]
    
    # Encontrar padrões temporais no conjunto de treino
    best_months = train_data.groupby('month')['spread'].mean().nlargest(3).index.tolist()
    best_days = train_data.groupby('dayofweek')['spread'].mean().nlargest(2).index.tolist()
    best_hours = train_data.groupby('hour')['spread'].mean().nlargest(4).index.tolist()
    
    logging.info(f"Meses com melhor spread: {best_months}")
    logging.info(f"Dias da semana com melhor spread: {best_days}")
    logging.info(f"Horas com melhor spread: {best_hours}")
    
    # Criar colunas de target inicializadas com zeros
    df['buy_target'] = 0
    df['sell_target'] = 0
    
    # Criar condições de compra
    buy_condition = (
        df['month'].isin(best_months) | 
        df['dayofweek'].isin(best_days) | 
        df['hour'].isin(best_hours)
    )
    
    # Aplicar targets apenas ao período de treino
    df.loc[train_mask & buy_condition, 'buy_target'] = 1
    df.loc[train_mask & ~buy_condition, 'sell_target'] = 1
    
    # Calcular estatísticas de distribuição
    train_buy_pct = df.loc[train_mask, 'buy_target'].mean() * 100
    test_buy_pct = df.loc[~train_mask, 'buy_target'].mean() * 100
    
    print(f"Buy target distribution - Train: {train_buy_pct:.2f}%, Test: {test_buy_pct:.2f}%")
    logging.info(f"✅ Buy target distribution - Train: {train_buy_pct:.2f}%, Test: {test_buy_pct:.2f}%")
    
    return df