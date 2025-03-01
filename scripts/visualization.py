import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
import logging
import seaborn as sns

def plot_strategy_results(df, save_path=None, show_plots=True):
    """
    Cria um conjunto completo de gráficos para analisar os resultados da estratégia.
    
    Parâmetros:
    - df: DataFrame com os resultados da estratégia
    - save_path: Caminho para salvar os gráficos (sem extensão)
    - show_plots: Se True, exibe os gráficos além de salvá-los
    """
    try:
        logging.info("📊 Gerando gráficos da estratégia...")
        
        # Verificar se o DataFrame tem as colunas necessárias
        required_columns = ['spread', 'signal', 'cum_strategy', 'total_value', 'fees']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"❌ ERRO: As colunas {missing_columns} estão ausentes no DataFrame.")
            return
            
        # Configurações gerais para os gráficos
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("dark")
        
        # 1. Gráfico do Spread com Sinais de Compra e Venda
        plt.figure(figsize=(16, 8))
        plt.plot(df.index, df['spread'], color='blue', alpha=0.7, linewidth=1.5)
        
        # Marcar sinais de compra
        buy_signals = df[df['signal'] == 1]
        plt.scatter(buy_signals.index, buy_signals['spread'], 
                   color='green', s=70, alpha=1, marker='^', label='Compra')
        
        # Marcar sinais de venda
        sell_signals = df[df['signal'] == -1]
        plt.scatter(sell_signals.index, sell_signals['spread'], 
                   color='red', s=70, alpha=1, marker='v', label='Venda')
        
        plt.title('Spread com Sinais de Negociação', fontsize=16)
        plt.ylabel('Spread (XAUUSD - PAXG)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Formatar eixo x para datas mais legíveis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(f"{save_path}_spread_signals.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        # 2. Gráfico de Retornos Acumulados
        plt.figure(figsize=(16, 8))
        
        # Verificar se temos retornos acumulados
        if 'cum_strategy' in df.columns:
            plt.plot(df.index, df['cum_strategy'], color='blue', linewidth=2, 
                     label='Retorno Acumulado')
            
            # Adicionar linha de referência (sem ganho/perda)
            plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Destacar períodos de drawdown
            rolling_max = df['cum_strategy'].cummax()
            drawdown = (df['cum_strategy'] / rolling_max - 1)
            plt.fill_between(df.index, df['cum_strategy'], rolling_max, 
                             where=df['cum_strategy'] < rolling_max, 
                             color='red', alpha=0.3, label='Drawdown')
            
            # Calcular métricas
            total_return = df['cum_strategy'].iloc[-1] - 1
            max_dd = drawdown.min()
            
            plt.title(f'Retorno Acumulado: {total_return:.2%}, Max Drawdown: {max_dd:.2%}', fontsize=16)
            plt.ylabel('Retorno Acumulado', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            # Formatar eixo x para datas mais legíveis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.gcf().autofmt_xdate()
            
            # Formatar eixo y como percentual
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}' if x >= 1 else f'{(x-1):.0%}'))
            
            if save_path:
                plt.savefig(f"{save_path}_cumulative_returns.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # 3. Gráfico de Gastos com Taxas (Fees)
        if 'fees' in df.columns:
            plt.figure(figsize=(16, 6))
            
            # Calcular fees acumulados
            fees_series = df['fees'].fillna(0)
            cumulative_fees = fees_series.cumsum()
            
            plt.plot(df.index, cumulative_fees, color='red', linewidth=2)
            plt.title(f'Taxas Acumuladas (Total: {cumulative_fees.iloc[-1]:.2f})', fontsize=16)
            plt.ylabel('Taxas Acumuladas', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Adicionar barras verticais quando ocorreram trades
            trade_days = df[df['signal'] != 0].index
            if len(trade_days) > 0:
                for day in trade_days:
                    plt.axvline(x=day, color='gray', linestyle='-', alpha=0.2)
            
            # Formatar eixo x para datas mais legíveis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.gcf().autofmt_xdate()
            
            if save_path:
                plt.savefig(f"{save_path}_fees.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        # 4. Gráfico da Evolução do Valor Total do Portfólio
        if 'total_value' in df.columns:
            plt.figure(figsize=(16, 8))
            
            plt.plot(df.index, df['total_value'], color='green', linewidth=2)
            plt.title('Evolução do Valor Total do Portfólio', fontsize=16)
            plt.ylabel('Valor ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Adicionar marcadores nos pontos de negociação
            trades = df[df['signal'] != 0]
            if not trades.empty:
                plt.scatter(trades.index, trades['total_value'], 
                           color='purple', s=50, alpha=0.7, 
                           marker='o', label='Negociação')
                plt.legend(fontsize=12)
            
            # Formatar eixo x para datas mais legíveis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.gcf().autofmt_xdate()
            
            if save_path:
                plt.savefig(f"{save_path}_portfolio_value.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        # 5. Comparação Buy vs. Sell Signals (Lucratividade)
        if 'signal' in df.columns and 'strategy_ret' in df.columns:
            plt.figure(figsize=(10, 6))
            
            buy_returns = df[df['signal'] == 1]['strategy_ret'].dropna()
            sell_returns = df[df['signal'] == -1]['strategy_ret'].dropna()
            
            # Criar boxplot comparando retornos de compra e venda
            data = [buy_returns, sell_returns]
            labels = ['Compra', 'Venda']
            
            plt.boxplot(data, labels=labels, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='blue'),
                      whiskerprops=dict(color='blue'),
                      capprops=dict(color='blue'),
                      medianprops=dict(color='red', linewidth=2))
            
            plt.title('Comparação de Retornos: Compra vs. Venda', fontsize=16)
            plt.ylabel('Retorno por Operação', fontsize=12)
            plt.grid(True, axis='y', alpha=0.3)
            
            # Adicionar estatísticas como texto
            if len(buy_returns) > 0:
                plt.text(0.85, 0.95, f'Compras: {len(buy_returns)}\nMédia: {buy_returns.mean():.2%}',
                        transform=plt.gca().transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if len(sell_returns) > 0:
                plt.text(0.85, 0.80, f'Vendas: {len(sell_returns)}\nMédia: {sell_returns.mean():.2%}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            if save_path:
                plt.savefig(f"{save_path}_signal_returns.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        logging.info(f"✅ Gráficos gerados com sucesso!")
        if save_path:
            logging.info(f"💾 Gráficos salvos em: {save_path}_*.png")
            
    except Exception as e:
        logging.error(f"❌ Erro ao gerar gráficos: {str(e)}")
        import traceback
        traceback.print_exc()