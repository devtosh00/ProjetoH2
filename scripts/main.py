import os
import traceback
import pandas as pd
import logging
import numpy as np
import joblib
import datetime

# Importações locais
from data_loader import load_and_process_data
from feature_engineering import add_indicators, create_gold_target
from model_training import train_dual_models
from backtest import (
    generate_dual_signals,
    compute_returns,
    compute_risk_metrics,
    generate_trading_log
)
from visualization import plot_strategy_results

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("strategy_logs.log"),
        logging.StreamHandler()
    ]
)

def ensure_directory(path):
    """Cria diretório se não existir."""
    os.makedirs(path, exist_ok=True)

def main():
    """
    Pipeline principal da estratégia quantitativa de trading de ouro.
    Executa todas as etapas desde o carregamento de dados até a avaliação final.
    """
    start_time = datetime.datetime.now()

    # Definir caminhos principais
    base_folder = r"C:\Users\samue\Documents\Projeto_Quant_H2\bases"
    models_folder = r"C:\Users\samue\Documents\Projeto_Quant_H2\models"
    results_folder = r"C:\Users\samue\Documents\Projeto_Quant_H2\results"
    ensure_directory(models_folder)
    ensure_directory(results_folder)

    # Período da estratégia 
    strategy_start_date = "2022-01-01"
    strategy_end_date = "2025-01-01"

    # Definição do perfil de risco
    risk_profile = 'moderate'  # Opções: 'conservative', 'moderate', 'aggressive'
    logging.info(f"⚙️ Configurando estratégia com perfil de risco: {risk_profile}")

    print("\n🚀 Iniciando pipeline de estratégia com modelos duais...\n")

    try:
        # 1️⃣ Carregar dados
        logging.info("📥 Carregando dados...")
        df = load_and_process_data(base_folder)
        if df is None or df.empty:
            raise ValueError("❌ ERRO: DataFrame de dados está vazio!")

        # 2️⃣ Adicionar indicadores técnicos (features)
        logging.info("📊 Adicionando indicadores técnicos...")
        df = add_indicators(df)

        # 3️⃣ Aplicar função de criação de target
        logging.info("🎯 Criando targets para modelo adaptativo...")
        df = create_gold_target(df)
        logging.info("✅ Target de compra e venda criado com sucesso.")

        # Salvar checkpoint após processamento de dados
        checkpoint_path = os.path.join(results_folder, "data_processed_checkpoint.pkl")
        joblib.dump(df, checkpoint_path)
        logging.info(f"🔄 Checkpoint salvo: {checkpoint_path}")

        # 4️⃣ Definir features para compra (X) e venda (Y)
        buy_features = [
            "spread", "spread_std", "spread_std_ma", "trend", "trend_ma",
            "upper_band", "middle_band", "lower_band", "atr",
            "macd_x", "macd_signal_x", "macd_hist_x",
            "stoch_k_x", "stoch_d_x",
            "rsi_x", "adx_x", "atr_x"
        ]

        sell_features = [
            "spread", "spread_std", "spread_std_ma", "trend", "trend_ma",
            "upper_band", "middle_band", "lower_band", "atr",
            "macd_y", "macd_signal_y", "macd_hist_y",
            "stoch_k_y", "stoch_d_y",
            "rsi_y", "adx_y", "atr_y"
        ]

        # Adicionar médias móveis do spread
        for window in [24, 7 * 24, 14 * 24, 30 * 24, 60 * 24]:
            buy_features.append(f"spread_ma_{window}")
            sell_features.append(f"spread_ma_{window}")

        # Filtrar e ordenar as features disponíveis
        buy_features = sorted([f for f in buy_features if f in df.columns])
        sell_features = sorted([f for f in sell_features if f in df.columns])
        logging.info(f"📌 Features disponíveis para treino de compra: {len(buy_features)}")
        logging.info(f"📌 Features disponíveis para treino de venda: {len(sell_features)}")

        # 5️⃣ Treinar modelos duais otimizados
        # Usar nomes específicos para os modelos
        buy_model_path = os.path.join(models_folder, "buy_model_optimized.pkl") 
        sell_model_path = os.path.join(models_folder, "sell_model_optimized.pkl")

        logging.info("🛠️ Treinando novos modelos XGBoost...")
        buy_model, sell_model = train_dual_models(
            df, buy_features, sell_features, optimize_hyperparams=False
        )
        joblib.dump(buy_model, buy_model_path)
        joblib.dump(sell_model, sell_model_path)
        logging.info("💾 Modelos treinados e salvos.")

        # 6️⃣ Aplicar filtro de data para backtest
        logging.info("🔍 Filtrando dados para o período de backtest...")
        df_filtered = df.loc[strategy_start_date:strategy_end_date].copy()
        logging.info(f"✅ Dados filtrados: {len(df_filtered)} registros.")

        # Garantir que todas as features necessárias estão disponíveis
        expected_buy_features = sorted(buy_model.get_booster().feature_names)
        expected_sell_features = sorted(sell_model.get_booster().feature_names)

        # Verificar features ausentes e preencher se necessário
        missing_buy_features = [f for f in expected_buy_features if f not in df_filtered.columns]
        if missing_buy_features:
            logging.warning(f"⚠️ Features de compra ausentes: {missing_buy_features}")
            for feature in missing_buy_features:
                df_filtered[feature] = 0.0

        missing_sell_features = [f for f in expected_sell_features if f not in df_filtered.columns]
        if missing_sell_features:
            logging.warning(f"⚠️ Features de venda ausentes: {missing_sell_features}")
            for feature in missing_sell_features:
                df_filtered[feature] = 0.0

        # Preencher valores ausentes
        df_filtered.fillna(df_filtered.median(), inplace=True)
        
        # 7️⃣ Gerar sinais de compra e venda
        logging.info("📈 Gerando sinais de compra e venda...")
        df_filtered = generate_dual_signals(
            df_filtered, buy_model, sell_model, 
            expected_buy_features, expected_sell_features
        )
        
        # Verificar se coluna 'signal' foi criada
        if 'signal' not in df_filtered.columns:
            if 'trade_signal' in df_filtered.columns:
                df_filtered['signal'] = df_filtered['trade_signal']
                logging.info("✅ Coluna 'signal' criada com base em 'trade_signal'")
            else:
                raise ValueError("❌ Nenhuma coluna de sinal foi gerada!")
        
        # Contagem de sinais
        total_signals = df_filtered['signal'].abs().sum()
        
        # 8️⃣ Calcular retornos e custos de transação
        logging.info("💰 Calculando retornos da estratégia...")
        transaction_cost = 0.0007  # 0.05% por operação
        df_filtered, returns_metrics = compute_returns(
            df_filtered, 
            transaction_cost=transaction_cost, 
            initial_capital=100000,
            risk_profile=risk_profile  # Usando o perfil de risco configurado
        )
        
        # 9️⃣ Calcular métricas de risco avançadas
        logging.info("📊 Calculando métricas de risco...")
        risk_metrics = compute_risk_metrics(df_filtered)
        
        # Extrair fees e outras métricas do returns_metrics
        total_fees = returns_metrics.get('total_fees', 0.0)
        trades_executed = returns_metrics.get('trades_executed', 0)
        avg_fee_per_trade = total_fees / trades_executed if trades_executed > 0 else 0
        initial_capital = 100000  # Capital inicial padrão
        fees_impact = total_fees / initial_capital
        
        # Alertas sobre desempenho
        if risk_metrics.get('Sharpe Ratio', 0) < 0.5:
            logging.warning("⚠️ Sharpe Ratio baixo: Estratégia pode não ser eficiente.")
        if returns_metrics.get('total_return', 0) < 0:
            logging.warning("⚠️ Retorno total negativo: Estratégia não rentável.")
        
        # Análise de custos operacionais
        logging.info("\n" + "=" * 70)
        logging.info(f"💰 ANÁLISE DE CUSTOS OPERACIONAIS 💰 (PERFIL: {risk_profile.upper()})")
        logging.info("=" * 70)
        logging.info(f"Total de fees: ${total_fees:.2f}")
        logging.info(f"Média de fee por trade: ${avg_fee_per_trade:.2f}")
        logging.info(f"Impacto dos fees no capital inicial: {fees_impact:.2%}")
        logging.info(f"Ignorados: {returns_metrics.get('ignored_sell_signals', 0)} sinais de venda em dias não úteis")
        logging.info(f"Bloqueados por risco: {returns_metrics.get('ignored_risk_signals', 0)} sinais (volatilidade/drawdown)")
        if 'fees_vs_return' in returns_metrics and returns_metrics.get('fees_vs_return', float('inf')) < float('inf'):
            logging.info(f"⚖️ Taxas em proporção ao retorno: {returns_metrics.get('fees_vs_return', 0):.2%}")
        logging.info("=" * 70 + "\n")
            
        # 🔟 Gerar log de operações e salvar resultados
        log_path = os.path.join(results_folder, "trade_log.csv")
        trade_log_df = generate_trading_log(df_filtered, log_path)
        logging.info(f"📝 Log de operações salvo: {log_path}")

        # 1️⃣1️⃣ Gerar gráficos da estratégia
        logging.info("📊 Gerando gráficos da estratégia...")
        plot_save_path = os.path.join(results_folder, f"strategy_results_{risk_profile}")
        plot_strategy_results(df_filtered, save_path=plot_save_path, show_plots=False)
        logging.info(f"📊 Gráficos salvos em: {plot_save_path}_*.png")

        # 1️⃣2️⃣ Salvar DataFrame final com resultados
        results_path = os.path.join(results_folder, f"strategy_results_{risk_profile}.csv")
        df_filtered.to_csv(results_path)
        logging.info(f"💾 Resultados completos salvos: {results_path}")

        # Resumo final com todas as métricas importantes
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60.0

        # 1️⃣3️⃣ Apresentar resumo completo
        logging.info("\n" + "=" * 70)
        logging.info(f"📊 RESUMO FINAL - ESTRATÉGIA COM PERFIL {risk_profile.upper()} 📊")
        logging.info("=" * 70)
        logging.info(f"⏱️ Tempo de execução: {execution_time:.2f} min")
        logging.info(f"📅 Período analisado: {strategy_start_date} até {strategy_end_date}")
        logging.info(f"📈 Total de sinais gerados: {total_signals}, Executados: {returns_metrics.get('trades_executed', 0)}")
        
        # Métricas de retorno
        logging.info("\n💰 MÉTRICAS DE RETORNO")
        logging.info(f"💰 Retorno Acumulado: {returns_metrics.get('total_return', 0):.2%}")
        logging.info(f"📈 Retorno Anualizado: {returns_metrics.get('annualized_return', 0):.2%}")
        logging.info(f"📊 Win Rate: {returns_metrics.get('win_rate', 0):.2%}")
        logging.info(f"📝 Trades Lucrativos: {returns_metrics.get('profit_trades', 0)}")
        logging.info(f"📝 Trades Perdedores: {returns_metrics.get('losing_trades', 0)}")
        logging.info(f"💵 Total de Fees: ${total_fees:.2f} ({fees_impact:.2%} do capital)")
        
        # Métricas de risco básicas
        logging.info("\n📉 MÉTRICAS DE RISCO BÁSICAS")
        logging.info(f"📉 Máximo Drawdown: {risk_metrics.get('Max Drawdown', 0):.2%}")
        logging.info(f"📊 Sharpe Ratio: {risk_metrics.get('Sharpe Ratio', 0):.2f}")
        logging.info(f"📊 Sortino Ratio: {risk_metrics.get('Sortino Ratio', 0):.2f}")
        logging.info(f"📊 Calmar Ratio: {risk_metrics.get('Calmar Ratio', 0):.2f}")
        logging.info(f"📊 VaR 95%: {risk_metrics.get('VaR 95%', 0):.2%}")
        logging.info(f"📊 CVaR 95%: {risk_metrics.get('CVaR 95%', 0):.2%}")
        
        # Métricas de risco avançadas
        logging.info("\n📊 MÉTRICAS DE RISCO AVANÇADAS")
        logging.info(f"📊 Gain to Pain Ratio: {risk_metrics.get('Gain to Pain Ratio', 0):.2f}")
        logging.info(f"📊 Omega Ratio: {risk_metrics.get('Omega Ratio', 0):.2f}")
        logging.info(f"📉 Ulcer Index: {risk_metrics.get('Ulcer Index', 0):.4f}")
        logging.info(f"📊 Sterling Ratio: {risk_metrics.get('Sterling Ratio', 0):.2f}")
        logging.info(f"📊 Skewness: {risk_metrics.get('Skewness', 0):.2f}")
        logging.info(f"📊 Max Consecutive Losses: {risk_metrics.get('Max Consecutive Losses', 0)}")
        logging.info(f"📊 Positive Months %: {risk_metrics.get('Positive Months %', 0):.1f}%")
        logging.info("=" * 70 + "\n")

        print(f"\n✅ Pipeline finalizado com sucesso! (Perfil: {risk_profile.upper()})\n")

    except Exception as e:
        logging.error(f"❌ Erro no pipeline: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()