import numpy as np

def generate_classic_signal(df, delta=1.0, trend_filter=None):
    """
    Gera sinal clássico: se avg_etf < XAUUSD - delta e, opcionalmente, se trend_filter == 1, sinal = 1; senão 0.
    """
    if trend_filter is not None:
        signal = np.where((df["avg_etf"] < df["XAUUSD"] - delta) & (trend_filter == 1), 1, 0)
    else:
        signal = np.where(df["avg_etf"] < df["XAUUSD"] - delta, 1, 0)
    return signal

def calculate_strategy_returns(df, cost=0.0007):
    """
    Calcula os retornos da estratégia:
      - Se o sinal do dia anterior for 1, retorno = (paxg_ret - xauusd_ret)
      - Caso contrário, retorna a taxa risk_free.
    Aplica custos de transação se houver mudança de posição.
    """
    # Sinal de trade: verifica mudança de posição
    df["trade"] = df["signal"] != df["signal"].shift(1)
    df["cost"] = np.where(df["trade"], cost, 0)
    
    strategy_ret = np.where(
        df["signal"].shift(1) == 1,
        df["paxg_ret"] - df["xauusd_ret"] - df["cost"],
        df["risk_free"] - df["cost"]
    )
    df["strategy_ret"] = strategy_ret
    df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    return df

def calculate_ml_strategy_returns(df, cost=0.0007):
    """
    Calcula os retornos da estratégia híbrida ML.
    Utiliza a coluna 'ml_position' para definir posições.
    """
    df["ml_trade"] = df["ml_position"] != df["ml_position"].shift(1)
    df["ml_cost"] = np.where(df["ml_trade"], cost, 0)
    ml_strategy_ret = np.where(
        df["ml_position"].shift(1) == 1,
        df["paxg_ret"] - df["xauusd_ret"] - df["ml_cost"],
        df["risk_free"] - df["ml_cost"]
    )
    df["ml_strategy_ret"] = ml_strategy_ret
    df["cum_ml_strategy"] = (1 + df["ml_strategy_ret"].fillna(0)).cumprod()
    return df
