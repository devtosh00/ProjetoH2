import pandas as pd
import numpy as np
import logging
import os
import itertools
import traceback
from pandas.api.types import is_numeric_dtype

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data(df):
    """
    Limpa os dados de entrada, removendo ou preenchendo valores nan e inf.
    Performance otimizada para grandes datasets.
    """
    logging.info("🧹 Limpando dados...")
    
    # Substituir inf e -inf apenas nas colunas numéricas para otimizar desempenho
    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Log das colunas com NaN antes de preencher
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logging.info(f"⚠️ Colunas com NaN antes da limpeza: {nan_cols}")
    
    # Preencher valores ausentes (otimizado para grandes datasets)
    if len(nan_cols) > 0:
        for col in nan_cols:
            df[col] = df[col].ffill().bfill()
    
    # Verificar se ainda há NaNs após o preenchimento
    remaining_nans = df.isna().sum().sum()
    if (remaining_nans > 0):
        logging.warning(f"⚠️ Existem {remaining_nans} valores NaN restantes após o preenchimento.")
    
    return df

def compute_returns(df, transaction_cost=0.0005, initial_capital=100000, risk_profile='moderate'):
    """
    Calcula os retornos da estratégia, com gestão de risco adaptativa e controle de taxas.
    
    Parâmetros:
    - df: DataFrame com os dados de mercado e sinais
    - transaction_cost: Custo por transação (%)
    - initial_capital: Capital inicial ($)
    - risk_profile: Perfil de risco ('aggressive', 'moderate', 'conservative')
    """
    try:
        logging.info(f"💰 Calculando retornos com perfil de risco: {risk_profile}...")
        df = df.copy()
        
        df = clean_data(df)  # Limpar dados de entrada

        if df.empty:
            raise ValueError("❌ ERRO: DataFrame está vazio após limpeza de dados.")

        # Verificar a existência de coluna de sinais - mais flexível quanto ao nome
        signal_col = None
        for col_name in ['trade_signal', 'signal']:
            if col_name in df.columns:
                signal_col = col_name
                break
                
        if signal_col is None:
            raise ValueError("❌ ERRO: Nenhuma coluna de sinal ('trade_signal' ou 'signal') foi encontrada")
            
        # Sempre criar coluna 'signal' para compatibilidade
        if signal_col != 'signal':
            df["signal"] = df[signal_col]
            logging.info(f"✅ Coluna 'signal' criada com base em '{signal_col}'")
        
        # Contar o número de trades potenciais
        buy_trades = len(df[df["signal"] == 1])
        sell_trades = len(df[df["signal"] == -1])
        total_trades = buy_trades + sell_trades
        logging.info(f"📊 Trades identificados: {buy_trades} compras, {sell_trades} vendas, total: {total_trades}")

        # Taxa livre de risco diária (2% ao ano convertida para taxa horária)
        risk_free_rate = 0.02 / (24 * 365)

        # Configurações baseadas no perfil de risco
        if risk_profile == 'conservative':
            position_size = 0.3         # 30% do capital disponível por operação
            volatility_threshold = 1.5  # Não operar quando volatilidade > 1.5x média
            stop_loss_pct = 0.03        # Stop loss de 3%
            take_profit_pct = 0.05      # Take profit de 5%
            max_drawdown_limit = 0.05   # Pausa temporária após drawdown de 5%
            fee_optimization = 0.75     # Reduz taxas em 25% para transações maiores
            
        elif risk_profile == 'moderate':
            position_size = 0.5         # 50% do capital disponível por operação
            volatility_threshold = 2.0  # Não operar quando volatilidade > 2x média
            stop_loss_pct = 0.05        # Stop loss de 5%
            take_profit_pct = 0.08      # Take profit de 8%
            max_drawdown_limit = 0.1    # Pausa temporária após drawdown de 10%
            fee_optimization = 0.85     # Reduz taxas em 15% para transações maiores
            
        else:  # 'aggressive'
            position_size = 0.8         # 80% do capital disponível por operação
            volatility_threshold = 3.0  # Não operar quando volatilidade > 3x média
            stop_loss_pct = 0.08        # Stop loss de 8%
            take_profit_pct = 0.12      # Take profit de 12%
            max_drawdown_limit = 0.15   # Pausa temporária após drawdown de 15%
            fee_optimization = 1.0      # Sem desconto em taxas
        
        # Calcular volatilidade e outras métricas de risco
        df['volatility'] = df['close_x'].pct_change().rolling(24).std() * np.sqrt(24)
        df['avg_volatility'] = df['volatility'].rolling(30 * 24).mean()
        df['volatility_ratio'] = df['volatility'] / df['avg_volatility'].replace(0, np.nan).fillna(1)
        
        # Converter colunas para float para evitar warnings
        df["cash"] = initial_capital
        df["paxg"] = 0.0
        df["xauusd"] = 0.0
        df["total_value"] = initial_capital
        df["fees"] = 0.0
        df["drawdown"] = 0.0
        df["stop_trading"] = False
        
        # Inicializar contadores e variáveis de rastreamento
        trades_executed = 0
        ignored_sell_signals = 0
        ignored_risk_signals = 0
        total_fees = 0.0
        peak_value = initial_capital
        consecutive_losses = 0
        
        # Preparar arrays numpy para operações mais rápidas
        signals = df["signal"].values
        is_business_day = np.ones(len(df), dtype=bool)
        
        # Verificar dias úteis se a coluna existir
        if "businessday" in df.columns:
            is_business_day = df["businessday"].values == 1
        
        # Loop principal - executar trades
        for i in range(1, len(df)):
            # Calcular drawdown atual e verificar limite
            current_value = df.at[df.index[i-1], "total_value"]
            peak_value = max(peak_value, current_value)
            current_drawdown = (current_value / peak_value - 1)
            df.at[df.index[i], "drawdown"] = current_drawdown
            
            # Verificar se deve pausar trading por excesso de drawdown
            if current_drawdown < -max_drawdown_limit:
                df.at[df.index[i], "stop_trading"] = True
                if not df.at[df.index[i-1], "stop_trading"]:
                    logging.info(f"⚠️ Trading pausado em {df.index[i]} devido a drawdown de {current_drawdown:.2%}")
            elif current_drawdown > -max_drawdown_limit/2 and df.at[df.index[i-1], "stop_trading"]:
                # Retomar trading após recuperação parcial
                df.at[df.index[i], "stop_trading"] = False
                logging.info(f"✅ Trading retomado em {df.index[i]} após drawdown reduzir para {current_drawdown:.2%}")
            else:
                df.at[df.index[i], "stop_trading"] = df.at[df.index[i-1], "stop_trading"]
            
            # Verificar condições de alta volatilidade
            high_volatility = False
            if i > 30*24 and 'volatility_ratio' in df.columns:  # Garantir que temos dados suficientes
                volatility_ratio = df.at[df.index[i], "volatility_ratio"]
                if not pd.isna(volatility_ratio) and volatility_ratio > volatility_threshold:
                    high_volatility = True
            
            # Ajustar tamanho da posição baseado em perdas consecutivas
            dynamic_position_size = position_size
            if consecutive_losses > 2:
                # Reduzir tamanho da posição após perdas consecutivas
                dynamic_position_size = position_size * (1 - min(0.5, consecutive_losses * 0.1))
            
            # Compra PAXG (com verificações de risco)
            if signals[i] == 1 and not df.at[df.index[i], "stop_trading"] and not high_volatility:
                # Determinar valor a investir
                available_cash = df.at[df.index[i-1], "cash"]
                invest_amount = available_cash * dynamic_position_size
                
                # Calcular quantidade e taxa - otimizada por tamanho da transação
                paxg_price = df.at[df.index[i], "close_x"]
                paxg_amount = invest_amount / paxg_price
                
                # Otimização de taxas baseada no tamanho da operação
                effective_fee_rate = transaction_cost
                if invest_amount > 10000:  # Transações grandes recebem desconto
                    effective_fee_rate *= fee_optimization
                
                fee = paxg_amount * paxg_price * effective_fee_rate
                
                # Atualizar posições e capital
                df.at[df.index[i], "paxg"] = df.at[df.index[i-1], "paxg"] + paxg_amount
                df.at[df.index[i], "cash"] = available_cash - (paxg_amount * paxg_price) - fee
                df.at[df.index[i], "xauusd"] = df.at[df.index[i-1], "xauusd"]
                df.at[df.index[i], "fees"] = df.at[df.index[i-1], "fees"] + fee
                
                # Rastrear fee total e incrementar contador
                total_fees += fee
                trades_executed += 1
                
                logging.debug(f"📈 COMPRA em {df.index[i]}: {paxg_amount:.2f} PAXG a ${paxg_price:.2f}, taxa: ${fee:.2f}")
            
            # Venda XAUUSD (com verificações de risco)
            elif signals[i] == -1:
                if is_business_day[i] and not df.at[df.index[i], "stop_trading"] and not high_volatility:
                    # Determinar valor a investir
                    available_cash = df.at[df.index[i-1], "cash"]
                    invest_amount = available_cash * dynamic_position_size
                    
                    # Calcular quantidade e taxa - otimizada por tamanho da transação
                    xauusd_price = df.at[df.index[i], "close_y"]
                    xauusd_amount = invest_amount / xauusd_price
                    
                    # Otimização de taxas baseada no tamanho da operação
                    effective_fee_rate = transaction_cost
                    if invest_amount > 10000:  # Transações grandes recebem desconto
                        effective_fee_rate *= fee_optimization
                    
                    fee = xauusd_amount * xauusd_price * effective_fee_rate
                    
                    # Atualizar posições e capital
                    df.at[df.index[i], "xauusd"] = df.at[df.index[i-1], "xauusd"] + xauusd_amount
                    df.at[df.index[i], "cash"] = available_cash - (xauusd_amount * xauusd_price) - fee
                    df.at[df.index[i], "paxg"] = df.at[df.index[i-1], "paxg"]
                    df.at[df.index[i], "fees"] = df.at[df.index[i-1], "fees"] + fee
                    
                    # Rastrear fee total e incrementar contador
                    total_fees += fee
                    trades_executed += 1
                    
                    logging.debug(f"📉 VENDA em {df.index[i]}: {xauusd_amount:.2f} XAUUSD a ${xauusd_price:.2f}, taxa: ${fee:.2f}")
                
                elif not is_business_day[i]:
                    # Se não é dia útil, manter posições e ignorar sinal
                    df.at[df.index[i], "paxg"] = df.at[df.index[i-1], "paxg"]
                    df.at[df.index[i], "cash"] = df.at[df.index[i-1], "cash"] * (1 + risk_free_rate)
                    df.at[df.index[i], "xauusd"] = df.at[df.index[i-1], "xauusd"]
                    df.at[df.index[i], "fees"] = df.at[df.index[i-1], "fees"]
                    
                    ignored_sell_signals += 1
                    logging.debug(f"⏸️ Sinal de VENDA ignorado em {df.index[i]}: Não é dia útil")
                
                else:
                    # Ignorado devido a outros fatores de risco (alta volatilidade ou stop trading)
                    df.at[df.index[i], "paxg"] = df.at[df.index[i-1], "paxg"]
                    df.at[df.index[i], "cash"] = df.at[df.index[i-1], "cash"] * (1 + risk_free_rate)
                    df.at[df.index[i], "xauusd"] = df.at[df.index[i-1], "xauusd"]
                    df.at[df.index[i], "fees"] = df.at[df.index[i-1], "fees"]
                    
                    ignored_risk_signals += 1
                    reason = "drawdown excessivo" if df.at[df.index[i], "stop_trading"] else "alta volatilidade"
                    logging.debug(f"⚠️ Sinal de VENDA ignorado em {df.index[i]} devido a {reason}")
            
            else:  # Sem operação
                df.at[df.index[i], "paxg"] = df.at[df.index[i-1], "paxg"]
                df.at[df.index[i], "xauusd"] = df.at[df.index[i-1], "xauusd"]
                df.at[df.index[i], "cash"] = df.at[df.index[i-1], "cash"] * (1 + risk_free_rate)
                df.at[df.index[i], "fees"] = df.at[df.index[i-1], "fees"]

            # Atualizar valor total do portfolio
            df.at[df.index[i], "total_value"] = (
                df.at[df.index[i], "cash"] + 
                df.at[df.index[i], "paxg"] * df.at[df.index[i], "close_x"] + 
                df.at[df.index[i], "xauusd"] * df.at[df.index[i], "close_y"]
            )
            
            # Atualizar contador de perdas consecutivas
            if i > 1:
                period_return = df.at[df.index[i], "total_value"] / df.at[df.index[i-1], "total_value"] - 1
                if period_return < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
        
        # Calcular retornos e retornos acumulados
        df["strategy_ret"] = df["total_value"].pct_change().fillna(0)
        df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()
        
        # Calcular métricas de retorno
        total_return = df["cum_strategy"].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (365 * 24 / len(df)) - 1
        daily_returns = df["strategy_ret"].dropna()
        
        # Cálculo do Sharpe Ratio
        if daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
            
        # Contagem de trades lucrativos e perdedores
        profit_trades = len(df[df["strategy_ret"] > 0])
        losing_trades = len(df[df["strategy_ret"] < 0])
        
        # Verificar consistência nos fees
        if abs(total_fees - df["fees"].iloc[-1]) > 0.01:
            logging.warning(f"⚠️ Divergência detectada no cálculo de fees: Calculado: {total_fees:.2f}, DataFrame: {df['fees'].iloc[-1]:.2f}")
        
        final_fees = df["fees"].iloc[-1]
        if final_fees < 0.01:  # Se for muito baixo, usar o total calculado
            final_fees = total_fees
        
        # Estatísticas de custo e impacto das taxas
        fee_per_trade = final_fees / trades_executed if trades_executed > 0 else 0
        fees_impact = final_fees / initial_capital
        fees_vs_return = final_fees / (total_return * initial_capital) if total_return > 0 else float('inf')
        
        # Estatísticas de risco
        max_drawdown = df["drawdown"].min()
        
        # Criar relatório detalhado de taxas
        logging.info(f"\n{'='*40}\n📊 RELATÓRIO DE TAXAS E RISCO ({risk_profile.upper()})\n{'='*40}")
        logging.info(f"💰 Total de taxas: ${final_fees:.2f}")
        logging.info(f"💸 Taxa média por operação: ${fee_per_trade:.2f}")
        logging.info(f"📉 Impacto das taxas no capital: {fees_impact:.2%}")
        if fees_vs_return < float('inf'):
            logging.info(f"⚖️ Taxas em relação ao retorno: {fees_vs_return:.2%}")
        logging.info(f"📊 Sinais originais: {total_trades}, Trades executados: {trades_executed}")
        logging.info(f"🛑 Sinais bloqueados: {ignored_sell_signals + ignored_risk_signals}")
        logging.info(f"📉 Máximo Drawdown: {max_drawdown:.2%}")
        
        return df, {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "trades_executed": trades_executed,
            "profit_trades": profit_trades,
            "losing_trades": losing_trades,
            "win_rate": profit_trades / trades_executed if trades_executed > 0 else 0,
            "ignored_sell_signals": ignored_sell_signals,
            "ignored_risk_signals": ignored_risk_signals,
            "total_fees": final_fees,
            "avg_fee_per_trade": fee_per_trade,
            "fees_impact": fees_impact,
            "fees_vs_return": fees_vs_return,
            "max_drawdown": max_drawdown,
            "risk_profile": risk_profile
        }
        
    except Exception as e:
        logging.error(f"❌ Erro ao calcular retornos: {e}")
        traceback.print_exc()
        raise

def compute_risk_metrics(df):
    """
    Calcula métricas de risco avançadas para a estratégia.
    
    Parâmetros:
    - df: DataFrame com os resultados da estratégia (deve incluir 'cum_strategy', 'strategy_ret')
    
    Retorna:
    - dict: Dicionário com as métricas de risco calculadas
    """
    try:
        logging.info("📊 Calculando métricas de risco avançadas...")
        df = df.copy()
        
        # Limpar dados
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        
        # Verificar se as colunas necessárias existem
        if "cum_strategy" not in df.columns:
            raise ValueError("❌ Coluna 'cum_strategy' não encontrada no DataFrame")
            
        if "strategy_ret" not in df.columns:
            raise ValueError("❌ Coluna 'strategy_ret' não encontrada no DataFrame")

        # --- Métricas Básicas de Risco ---
        
        # Máximo Drawdown
        rolling_max = df["cum_strategy"].cummax()
        drawdown = (df["cum_strategy"] / rolling_max - 1)
        max_drawdown = drawdown.min()

        # Calmar Ratio (annualized return / max drawdown)
        annualized_return = (1 + df["cum_strategy"].iloc[-1] - 1) ** (365 * 24 / len(df)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

        # Value at Risk (VaR)
        var_95 = np.percentile(df["strategy_ret"].dropna(), 5)

        # Conditional Value at Risk (CVaR) / Expected Shortfall
        cvar_95 = df[df["strategy_ret"] <= var_95]["strategy_ret"].mean()
        
        # Volatility (annualized)
        volatility = df["strategy_ret"].std() * np.sqrt(365 * 24) if df["strategy_ret"].std() > 0 else 0
        
        # Sharpe Ratio
        risk_free = 0.02 / 365  # Taxa livre de risco diária (2% a.a.)
        excess_return = df["strategy_ret"] - risk_free
        sharpe_ratio = excess_return.mean() / volatility if volatility > 0 else 0
        
        # Sortino Ratio (considerando apenas retornos negativos)
        downside_returns = df["strategy_ret"][df["strategy_ret"] < 0]
        downside_deviation = downside_returns.std() * np.sqrt(365 * 24) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # --- Métricas Avançadas de Risco ---
        
        # 1. Gain to Pain Ratio
        gains = df["strategy_ret"][df["strategy_ret"] > 0].sum()
        pains = abs(df["strategy_ret"][df["strategy_ret"] < 0].sum())
        gain_to_pain = gains / pains if pains > 0 else float('inf')
        
        # 2. Maximum Consecutive Losses
        ret_binary = (df["strategy_ret"] < 0).astype(int).values
        max_consecutive_losses = max(
            [sum(1 for _ in group) for key, group in itertools.groupby(ret_binary) if key == 1], 
            default=0
        )
        
        # 3. Ulcer Index (raiz quadrada da média dos quadrados dos drawdowns)
        ulcer_index = np.sqrt((drawdown.clip(upper=0) ** 2).mean())
        
        # 4. Maximum Time Under Water
        drawdown_periods = drawdown < 0
        current_underwater = 0
        max_underwater = 0
        for is_dd in drawdown_periods:
            if is_dd:
                current_underwater += 1
                max_underwater = max(max_underwater, current_underwater)
            else:
                current_underwater = 0
        
        # 5. Skewness e Kurtosis da distribuição de retornos
        skewness = df["strategy_ret"].skew()
        kurtosis = df["strategy_ret"].kurtosis()
        
        # 6. Média e Mediana de Retornos Mensais
        df['month'] = df.index.to_period('M')
        monthly_returns = df.groupby('month')['strategy_ret'].sum()
        mean_monthly_return = monthly_returns.mean()
        median_monthly_return = monthly_returns.median()
        
        # 7. Percentual de Meses Positivos
        positive_months = (monthly_returns > 0).mean() * 100
        
        # 8. Razão de Omega (probabilidade ponderada de ganhos versus perdas)
        threshold = 0
        returns_above_threshold = df["strategy_ret"][df["strategy_ret"] > threshold].sum()
        returns_below_threshold = abs(df["strategy_ret"][df["strategy_ret"] < threshold].sum())
        omega_ratio = returns_above_threshold / returns_below_threshold if returns_below_threshold > 0 else float('inf')
        
        # 9. Desvio-padrão da sequência de drawdowns (mede a consistência dos drawdowns)
        drawdown_sequence = drawdown[drawdown < -0.01].dropna()
        drawdown_std = drawdown_sequence.std() if len(drawdown_sequence) > 0 else 0
        
        # 10. Sterling Ratio (retorno anualizado / drawdown médio)
        avg_drawdown = drawdown.mean() if len(drawdown) > 0 else 0
        sterling_ratio = annualized_return / abs(avg_drawdown) if avg_drawdown < 0 else float('inf')

        # Log das métricas principais
        logging.info(f"✅ Métricas de risco calculadas com sucesso!")
        logging.info(f"📉 Max Drawdown: {max_drawdown:.4f}")
        logging.info(f"📈 Calmar Ratio: {calmar_ratio:.4f}")
        logging.info(f"⚠️ VaR 95%: {var_95:.4f}")
        logging.info(f"⚠️ CVaR 95%: {cvar_95:.4f}")
        logging.info(f"📊 Volatilidade anualizada: {volatility:.4f}")
        logging.info(f"📊 Skewness: {skewness:.4f} (>0 é favorável)")
        logging.info(f"📊 Gain to Pain: {gain_to_pain:.4f}")
        logging.info(f"📊 Percentual de meses positivos: {positive_months:.2f}%")

        # Retornar todas as métricas calculadas
        return {
            # Métricas básicas
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            "Volatilidade": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            
            # Métricas avançadas
            "Gain to Pain Ratio": gain_to_pain,
            "Max Consecutive Losses": max_consecutive_losses,
            "Ulcer Index": ulcer_index,
            "Maximum Time Under Water": max_underwater,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Mean Monthly Return": mean_monthly_return,
            "Median Monthly Return": median_monthly_return,
            "Positive Months %": positive_months,
            "Omega Ratio": omega_ratio,
            "Drawdown Std": drawdown_std,
            "Sterling Ratio": sterling_ratio
        }

    except Exception as e:
        logging.error(f"❌ Erro ao calcular métricas de risco: {e}")
        raise

def generate_dual_signals(df, buy_model, sell_model, buy_features, sell_features, buy_threshold=0.7, sell_threshold=0.7):
    """
    Gera sinais de compra e venda usando os modelos duais, permitindo compras a qualquer momento
    mas vendas apenas em dias úteis em horário comercial.
    """
    try:
        logging.info("📈 Gerando sinais usando modelos duais...")
        df = df.copy()
        transaction_cost = 0.0005  # Definir o custo de transação
        
        # Verificar features disponíveis
        available_buy_features = [f for f in buy_features if f in df.columns]
        available_sell_features = [f for f in sell_features if f in df.columns]
        if not available_buy_features or not available_sell_features:
            raise ValueError("❌ Nenhuma das features necessárias está disponível no DataFrame")
        
        logging.info(f"📋 Features para modelo de compra: {len(available_buy_features)}")
        logging.info(f"📋 Features para modelo de venda: {len(available_sell_features)}")
        
        # Extrair dados para predição e converter para numpy arrays
        buy_data = df[available_buy_features].values
        sell_data = df[available_sell_features].values
        
        # Gerar probabilidades de compra e venda usando arrays numpy
        df["buy_prob"] = buy_model.predict_proba(buy_data)[:, 1]
        df["sell_prob"] = sell_model.predict_proba(sell_data)[:, 1]
        
        # Inicializar colunas de tracking
        df["position"] = 0
        df["trade_signal"] = 0
        df["signal"] = 0  # Adicionar coluna signal para compatibilidade
        df["entry_price"] = 0.0
        df["gain"] = 0.0
        
        position = 0  # Rastreia a posição atual
        entry_price = 0.0  # Preço de entrada
        gain = 0.0  # Ganho acumulado
        
        for i in range(1, len(df)):
            # Sinal de compra (quando modelo de compra tem alta confiança) - permitido 24/7
            if df["buy_prob"].iloc[i] > buy_threshold and position == 0:
                df.loc[df.index[i], "trade_signal"] = 1
                df.loc[df.index[i], "signal"] = 1  # Atualizar ambas as colunas
                position = 1
                entry_price = df["close_x"].iloc[i] if "close_x" in df.columns else 0
                logging.debug(f"✅ Sinal de COMPRA gerado em {df.index[i]}, prob: {df['buy_prob'].iloc[i]:.2f}")
                
            # Sinal de venda (quando modelo de venda tem alta confiança)
            # Apenas gerar o sinal - a verificação de dia útil ocorre em compute_returns
            elif df["sell_prob"].iloc[i] > sell_threshold and position > 0:
                # Verificar se é dia útil (apenas para venda)
                is_business_day = True
                if "businessday" in df.columns:
                    is_business_day = df["businessday"].iloc[i] == 1
                
                if is_business_day:  # Apenas gerar sinal de venda em dias úteis
                    df.loc[df.index[i], "trade_signal"] = -1
                    df.loc[df.index[i], "signal"] = -1  # Atualizar ambas as colunas
                    
                    # Calcular ganho realizado
                    exit_price = df["close_x"].iloc[i] if "close_x" in df.columns else 0
                    realized_gain = exit_price - entry_price - (transaction_cost * 2)
                    gain += realized_gain
                    
                    position = 0  # Reset position
                    logging.debug(f"✅ Sinal de VENDA gerado em {df.index[i]}, prob: {df['sell_prob'].iloc[i]:.2f}")
                else:
                    logging.debug(f"🚫 Sinal de VENDA bloqueado em {df.index[i]}: Não é dia útil em horário comercial")
            
            # Atualizar posição e ganho
            df.loc[df.index[i], "position"] = position
            df.loc[df.index[i], "gain"] = gain
        
        buy_count = (df["trade_signal"] == 1).sum()
        sell_count = (df["trade_signal"] == -1).sum()
        logging.info(f"📊 Sinais gerados: {buy_count} compras, {sell_count} vendas")
        
        return df
    
    except Exception as e:
        logging.error(f"❌ Erro ao gerar sinais duais: {e}")
        raise

def generate_trading_log(df, file_path="C:/Users/samue/Documents/projeto Quant-H2/bases/trade_log.csv"):
    """
    Gera o log de operações a partir do DataFrame 'df' e salva em um arquivo CSV.
    """
    try:
        logging.info("📝 Criando trade log...")
        
        # Verificar qual coluna de sinal usar (mais flexível)
        signal_col = None
        for col_name in ['trade_signal', 'signal']:
            if col_name in df.columns:
                signal_col = col_name
                break
                
        if signal_col is None:
            raise ValueError("❌ ERRO: Nenhuma coluna de sinal ('trade_signal' ou 'signal') foi encontrada")
        
        # Filtrar apenas as linhas onde ocorreram operações - mais eficiente
        trades_df = df[df[signal_col] != 0].copy()
        
        if trades_df.empty:
            logging.warning("⚠️ Nenhuma operação encontrada para gerar o log de trades")
            return pd.DataFrame()
            
        # Verificar se existe a coluna businessday
        has_business_day = "businessday" in df.columns
        
        # Criar lista para trade_log
        trade_log = []
        
        for idx, row in trades_df.iterrows():
            # Buscar o valor anterior de fees de forma segura
            prev_fees = 0
            if "fees" in row and idx != df.index[0]:
                # Encontrar o índice anterior no DataFrame original
                prev_idx_pos = df.index.get_loc(idx) - 1
                if prev_idx_pos >= 0:
                    prev_fees = df["fees"].iloc[prev_idx_pos]
            
            trade_info = {
                "Timestamp": idx,
                "Tipo": "Compra" if row[signal_col] == 1 else "Venda",
                "PAXG_Preço": row["close_x"] if "close_x" in row else None,
                "XAUUSD_Preço": row["close_y"] if "close_y" in row else None,
                "Spread": row["spread"] if "spread" in row else None,
                "Probabilidade": row["buy_prob"] if row[signal_col] == 1 and "buy_prob" in row else 
                               (row["sell_prob"] if "sell_prob" in row else None),
                "Retorno": row["strategy_ret"] if "strategy_ret" in row else None,
                "Taxa": row["fees"] - prev_fees if "fees" in row else None
            }
            
            # Adicionar informação sobre dia útil se disponível
            if has_business_day:
                trade_info["Dia_Útil"] = "Sim" if row["businessday"] == 1 else "Não"
            
            trade_log.append(trade_info)

        # Criar DataFrame e exportar para CSV
        trade_log_df = pd.DataFrame(trade_log)
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        trade_log_df.to_csv(file_path, index=False)
        logging.info(f"💾 Trade log salvo em: {file_path}")
        logging.info(f"📊 Total de operações registradas: {len(trade_log_df)}")
        
        return trade_log_df
        
    except Exception as e:
        logging.error(f"❌ Erro ao criar ou salvar trade log: {e}")
        raise