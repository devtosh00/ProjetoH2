import numpy as np
import matplotlib.pyplot as plt

def calculate_performance_metrics(cum_returns, daily_returns, risk_free_rate, trades, days):
    years = days / 252
    annual_return = (cum_returns[-1] ** (1/years) - 1) * 100
    daily_vol = np.std(daily_returns)
    annual_vol = daily_vol * np.sqrt(252) * 100
    avg_rf = np.mean(risk_free_rate) * 252 * 100
    sharpe_ratio = (annual_return - avg_rf) / annual_vol if annual_vol > 0 else 0
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns / peak - 1) * 100
    max_drawdown = np.min(drawdown)
    return annual_return, annual_vol, sharpe_ratio, max_drawdown, trades / years

def plot_performance(df, cum_strategy, cum_ml_strategy, cum_xauusd, cum_riskfree):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, cum_strategy, label="Estratégia (Z-Score)", color="blue", linewidth=2)
    plt.plot(df.index, cum_ml_strategy, label="Estratégia ML Híbrida", color="purple", linewidth=2)
    plt.plot(df.index, cum_xauusd, label="Buy & Hold XAUUSD", linestyle=":", color="red")
    plt.plot(df.index, cum_riskfree, label="Taxa Risk-Free", linestyle="-.", color="gray")
    plt.xlabel("Data")
    plt.ylabel("Retorno Acumulado")
    plt.title("Comparação de Retornos Acumulados")
    plt.legend()
    plt.grid(True)
    plt.show()
