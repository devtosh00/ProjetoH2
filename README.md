Gold Trading Strategy with Dual ML Models
Overview
This repository implements a quantitative trading strategy for gold markets using machine learning models. The strategy employs separate models for buy and sell signals (dual model approach), allowing for asymmetric decision making optimized for each action. The system trades between PAXG (tokenized gold) and XAUUSD (traditional gold spot), capitalizing on market inefficiencies while implementing adaptive risk management techniques.

Project Structure

.
├── bases/                # Raw data files
│   ├── dados_com_spread.csv
│   ├── fred_interest_rates.csv
│   ├── PAXG_daily.csv
│   └── XAUUSD_daily.csv
├── models/              # Trained machine learning models
│   ├── buy_model.pkl
│   ├── buy_model_optimized.pkl
│   ├── sell_model.pkl
│   └── sell_model_overfitted.pkl
├── notebooks/           # Jupyter notebooks for extracts
├── results/             # Strategy results and visualizations
│   └── data_processed_checkpoint.pkl
├── scripts/             # Core implementation
│   ├── __init__.py
│   ├── backtest.py      # Backtesting engine
│   ├── data_loader.py   # Data processing utilities
│   ├── feature_engineering.py  # Technical indicators
│   ├── main.py          # Main execution script
│   ├── model_training.py  # ML model training
│   └── visualization.py # Results visualization
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── strategy_logs.log    # Execution logs





Methodology
The strategy follows a comprehensive pipeline:


Data Loading & Processing: Combines PAXG and XAUUSD price data with spread information
Feature Engineering: Creates technical indicators and specialized features
Model Training: Trains separate ML models for buy and sell decisions
Signal Generation: Generates trading signals based on model predictions
Risk Management: Applies risk filters based on market conditions and portfolio status
Backtesting: Simulates trading with realistic transaction costs and constraints
Performance Analysis: Calculates comprehensive performance and risk metrics


Risk Profiles

Profile	Position Size	Volatility Threshold	Stop Loss	Take Profit	Max Drawdown
Conservative	30%	1.5x	3%	5%	5%
Moderate	50%	2.0x	5%	8%	10%
Aggressive	80%	3.0x	8%	12%	15%



The strategy supports three risk profiles:

Profile	Position Size	Volatility Threshold	Stop Loss	Take Profit	Max Drawdown
Conservative	30%	1.5x	3%	5%	5%
Moderate	50%	2.0x	5%	8%	10%
Aggressive	80%	3.0x	8%	12%	15%
Risk Management Features
Adaptive Position Sizing: Reduces position size after consecutive losses
Volatility Filter: Avoids trading during periods of abnormal volatility
Drawdown Protection: Temporarily pauses trading after reaching maximum drawdown limit
Differential Trading Hours: Buy operations allowed 24/7, sell operations only during business days
Fee Optimization: Adjusts transaction costs based on trade size
Risk Metrics Explained
Basic Risk Metrics
Max Drawdown: Maximum peak-to-trough decline, measures worst-case scenario
Sharpe Ratio: Risk-adjusted return (annualized return / volatility)
Sortino Ratio: Similar to Sharpe but only penalizes downside volatility
Calmar Ratio: Annualized return / maximum drawdown
Value at Risk (VaR 95%): Maximum loss expected in 95% of cases
Conditional Value at Risk (CVaR 95%): Expected loss when losses exceed VaR
Advanced Risk Metrics
Gain to Pain Ratio: Sum of all positive returns / absolute sum of all negative returns
Maximum Consecutive Losses: Longest streak of losing trades
Ulcer Index: Square root of mean squared drawdown, measures drawdown severity
Maximum Time Under Water: Longest period (in hours) where portfolio value stayed below previous peak
Skewness: Asymmetry of return distribution (positive values are favorable)
Kurtosis: "Tailedness" of returns, higher values indicate more extreme outcomes
Omega Ratio: Probability-weighted ratio of gains to losses
Sterling Ratio: Annualized return / average drawdown
Positive Months %: Percentage of months with positive returns
Performance Metrics
Total Return: Overall percentage gain/loss
Annualized Return: Return normalized to a yearly basis
Win Rate: Percentage of profitable trades
Profitable Trades: Number of trades with positive returns
Losing Trades: Number of trades with negative returns
Fee Impact: Transaction costs as a percentage of initial capital
Monthly Return Statistics: Mean, median, and distribution of monthly returns
Transaction Cost Analysis
The strategy implements a detailed transaction cost model:

Base fee: 0.07% per transaction
Fee optimization for large trades based on risk profile
Detailed fee tracking and reporting
Analysis of fee impact on overall performance





Métricas de Risco para Estratégias Quantitativas
Este documento explica detalhadamente todas as métricas de risco implementadas no nosso sistema de trading quantitativo. Compreender estas métricas é essencial para avaliação e otimização de estratégias.

Métricas Básicas de Risco
1. Max Drawdown (Máximo Drawdown)
O que é: Maior perda percentual desde um pico até o vale subsequente.
Como interpretar: Quanto menor, melhor. Representa o pior cenário histórico de perda.
Valor típico: -5% a -25% para estratégias moderadas.
No código: max_drawdown = drawdown.min()
2. Sharpe Ratio
O que é: Retorno excedente (acima da taxa livre de risco) por unidade de risco (volatilidade).
Como interpretar: Maior é melhor. Avalia eficiência de risco-retorno.
Valor típico: >1 é bom, >2 é muito bom, >3 é excelente.
No código: sharpe_ratio = excess_return.mean() / volatility
3. Sortino Ratio
O que é: Similar ao Sharpe, mas considera apenas volatilidade negativa (downside risk).
Como interpretar: Maior é melhor. Foco em risco de queda.
Valor típico: >1 é bom, >2 é muito bom.
No código: sortino_ratio = annualized_return / downside_deviation
4. Calmar Ratio
O que é: Retorno anualizado dividido pelo máximo drawdown.
Como interpretar: Maior é melhor. Mede retorno por unidade de risco de drawdown.
Valor típico: >0.5 é aceitável, >1 é bom, >3 é excelente.
No código: calmar_ratio = annualized_return / abs(max_drawdown)
5. Value at Risk (VaR 95%)
O que é: Perda máxima esperada em 95% dos casos.
Como interpretar: Quanto mais próximo de zero, menor o risco de perdas extremas.
Valor típico: Idealmente não superior a -3% para estratégias conservadoras.
No código: var_95 = np.percentile(df["strategy_ret"].dropna(), 5)
6. Conditional Value at Risk (CVaR 95%)
O que é: Perda média esperada nos 5% piores casos (além do VaR).
Como interpretar: Mede o "tail risk" ou risco de cauda. Menor é melhor.
Valor típico: Idealmente não superior a -4% para estratégias conservadoras.
No código: [cvar_95 = df[df["strategy_ret"] <= var_95]["strategy_ret"].mean()](http://vscodecontentref/5)
7. Volatilidade (Anualizada)
O que é: Desvio padrão dos retornos, anualizado.
Como interpretar: Menor indica maior estabilidade.
Valor típico: 5%-15% para estratégias de médio risco.
No código: volatility = df["strategy_ret"].std() * np.sqrt(365 * 24)
Métricas Avançadas de Risco
8. Gain to Pain Ratio
O que é: Soma dos retornos positivos dividida pela soma absoluta dos retornos negativos.
Como interpretar: >1 significa mais ganhos que perdas. Maior é melhor.
Valor típico: >1.5 é bom, >2 é excelente.
No código: gain_to_pain = gains / pains
9. Maximum Consecutive Losses
O que é: Maior sequência de perdas consecutivas.
Como interpretar: Menor é melhor. Mede resiliência psicológica necessária.
Valor típico: <5 é confortável para maioria dos traders.
No código: Calculado via itertools.groupby dos retornos negativos
10. Ulcer Index
O que é: Raiz quadrada da média dos quadrados dos drawdowns.
Como interpretar: Menor é melhor. Penaliza drawdowns maiores e mais longos.
Valor típico: <1% é excelente, <3% é bom.
No código: ulcer_index = np.sqrt((drawdown.clip(upper=0) ** 2).mean())
11. Maximum Time Under Water
O que é: Maior período contínuo (em períodos de tempo) abaixo do pico anterior.
Como interpretar: Menor é melhor. Mede quanto tempo se leva para recuperar de perdas.
Valor típico: <100 horas é preferível para estratégias intradiárias.
No código: Calculado com contagem de períodos em drawdown
12. Skewness (Assimetria)
O que é: Medida da assimetria da distribuição de retornos.
Como interpretar: Positivo é melhor (mais retornos extremamente positivos).
Valor típico: >0 é desejável, >0.5 é excelente.
No código: skewness = df["strategy_ret"].skew()
13. Kurtosis (Curtose)
O que é: Medida da "cauda" da distribuição de retornos.
Como interpretar: Menor é melhor quando positivo. Alta curtose indica mais eventos extremos.
Valor típico: <3 indica menos eventos extremos que distribuição normal.
No código: kurtosis = df["strategy_ret"].kurtosis()
14. Positive Months %
O que é: Percentual de meses com retorno positivo.
Como interpretar: Maior é melhor. Mede consistência.
Valor típico: >60% é bom, >70% é excelente.
No código: positive_months = (monthly_returns > 0).mean() * 100
15. Omega Ratio
O que é: Probabilidade ponderada de ganhos versus perdas.
Como interpretar: Maior é melhor. >1 significa vantagem estatística.
Valor típico: >1.5 é bom, >2 é excelente.
No código: omega_ratio = returns_above_threshold / returns_below_threshold
16. Sterling Ratio
O que é: Retorno anualizado dividido pelo drawdown médio.
Como interpretar: Maior é melhor. Variação menos penalizante do Calmar.
Valor típico: >1 é bom, >2 é excelente.
No código: sterling_ratio = annualized_return / abs(avg_drawdown)
17. Drawdown Standard Deviation
O que é: Desvio padrão da amplitude dos drawdowns.
Como interpretar: Menor é melhor. Mede consistência dos drawdowns.
Valor típico: <0.02 para estratégias estáveis.
No código: drawdown_std = drawdown_sequence.std()
Métricas de Transação e Custos
18. Fee Impact
O que é: Impacto dos custos de transação no capital inicial.
Como interpretar: Menor é melhor. Mede a eficiência em termos de custo.
Valor típico: <1% para estratégias eficientes em custo.
No código: fees_impact = final_fees / initial_capital
19. Fees vs Return
O que é: Proporção entre taxas pagas e retorno gerado.
Como interpretar: Menor é melhor. Ideal abaixo de 20%.
Valor típico: <15% para estratégias eficientes.
No código: fees_vs_return = final_fees / (total_return * initial_capital)
20. Win Rate
O que é: Percentual de operações lucrativas.
Como interpretar: Maior é melhor, mas deve ser avaliado junto com profit factor.
Valor típico: >50% é desejável, >60% é excelente.
No código: win_rate = profit_trades / trades_executed
