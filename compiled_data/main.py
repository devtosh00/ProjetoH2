import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importa os módulos criados
from data_loader import load_all_data
from data_preprocessing import adjust_scales, align_data, create_unified_df
from feature_engineering import calculate_z_scores, calculate_trend
from model_training import train_lstm_model
from strategy import generate_classic_signal, calculate_strategy_returns, calculate_ml_strategy_returns
from performance import calculate_performance_metrics, plot_performance

# Configurações iniciais
base_folder = r"C:\Users\samue\Documents\projeto Quant-H2\compiled_data"
start_date = "2022-01-01"
end_date = "2025-01-01"

# Carregar os dados
gld, iau, paxg, xauusd, fred = load_all_data(base_folder, start_date, end_date)

# Ajustar escalas
gld, iau, paxg = adjust_scales(gld, iau, paxg)

# Alinhar dados
gld, iau, paxg, xauusd, fred_risk, common_dates = align_data(gld, iau, paxg, xauusd, fred, start_date, end_date)

# Criar DataFrame unificado
df = create_unified_df(gld, iau, paxg, xauusd, fred_risk)

# Calcular indicadores técnicos (z-scores, etc.)
df = calculate_z_scores(df)
xauusd_trend = calculate_trend(xauusd)
df["xauusd_trend"] = xauusd_trend.reindex(common_dates, method='ffill')

# Gerar sinal de arbitragem otimizado (usando, por exemplo, delta=1.0)
delta = 1.0
df["signal_classic"] = np.where((df["avg_etf"] < df["XAUUSD"] - delta) & (df["xauusd_trend"] == 1), 1, 0)
# Escolher o sinal para a estratégia
df["signal"] = df["signal_classic"]

# Cálculo dos retornos diários (open-to-close) para PAXG e XAUUSD
paxg["ret"] = (paxg["close"] / paxg["open"]) - 1
xauusd["ret"] = (xauusd["close"] / xauusd["open"]) - 1
df["paxg_ret"] = paxg["ret"].loc[df.index]
df["xauusd_ret"] = xauusd["ret"].loc[df.index]

# Calcular retornos da estratégia clássica
df = calculate_strategy_returns(df)

# (Opcional) Treinar modelo LSTM e obter previsões para integração futura
# lstm_model, predictions_inv, scaler = train_lstm_model(df["spread"])
# ... (poderia incorporar o sinal preditivo)

# Calcular retornos buy & hold para comparação
cum_paxg = (1 + paxg["ret"]).cumprod()
cum_xauusd = (1 + xauusd["ret"]).cumprod()
cum_riskfree = (1 + df["risk_free"]).cumprod()

# Visualizar resultados
plot_performance(df, df["cum_strategy"], df["cum_strategy"], 5cum_xauusd, cum_riskfree)

# Exibir resultados finais
print("Retorno acumulado da estratégia: {:.2f}%".format((df["cum_strategy"].iloc[-1]-1)*100))
