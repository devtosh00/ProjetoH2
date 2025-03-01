import os
import pandas as pd
import logging

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_process_data(base_folder):

    logging.info("🚀 Iniciando o carregamento dos dados...")

    # Caminhos dos arquivos
    paxg_file = os.path.join(base_folder, "PAXG_daily.csv")
    xauusd_file = os.path.join(base_folder, "XAUUSD_daily.csv")

    # Verificação da existência dos arquivos
    if not os.path.exists(paxg_file) or not os.path.exists(xauusd_file):
        raise FileNotFoundError(f"❌ Arquivo(s) CSV não encontrado(s)! Verifique os caminhos:\n   - {paxg_file}\n   - {xauusd_file}")

    # Carregamento dos arquivos CSV
    paxg = pd.read_csv(paxg_file, parse_dates=["Timestamp"])
    xauusd = pd.read_csv(xauusd_file, parse_dates=["Timestamp"])

    # Verifica se as colunas essenciais estão presentes
    required_columns = {"High", "Low", "Close"}
    for df_name, df in {"PAXG": paxg, "XAUUSD": xauusd}.items():
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"❌ Colunas ausentes no arquivo {df_name}: {missing_columns}. Verifique o formato do CSV!")

    # Renomeia colunas para evitar conflitos ao mesclar
    rename_map = {
        "High": "high",
        "Low": "low",
        "Close": "close"
    }
    paxg.rename(columns={col: f"{rename_map[col]}_x" for col in required_columns}, inplace=True)
    xauusd.rename(columns={col: f"{rename_map[col]}_y" for col in required_columns}, inplace=True)

    # Mescla os DataFrames usando a coluna Timestamp
    df = pd.merge(paxg, xauusd, on="Timestamp", how="inner")

    # Remove índices duplicados, se existirem
    if df.duplicated(subset=["Timestamp"]).any():
        logging.warning("⚠️ Índices duplicados encontrados. Removendo...")
        df.drop_duplicates(subset=["Timestamp"], keep="first", inplace=True)

    # Ordena os dados por Timestamp
    df.sort_values(by="Timestamp", inplace=True)

    # Adicionar coluna de dia útil (segunda a sexta) e horário comercial (9h-18h)
    # Dia da semana: 0=Segunda, 1=Terça, 2=Quarta, 3=Quinta, 4=Sexta, 5=Sábado, 6=Domingo
    df['businessday'] = ((df['Timestamp'].dt.dayofweek <= 4) &  # Segunda a Sexta
                         (df['Timestamp'].dt.hour >= 9) &      # Depois das 9h
                         (df['Timestamp'].dt.hour < 18)).astype(int)  # Antes das 18h
    
    # Contar dias úteis vs não úteis
    business_count = df['businessday'].sum()
    non_business_count = len(df) - business_count
    business_pct = business_count / len(df) * 100
    
    logging.info(f"📊 Dias úteis em horário comercial: {business_count} ({business_pct:.1f}%)")
    logging.info(f"📊 Fora do horário comercial/fim de semana: {non_business_count}")
    
    # Define Timestamp como índice para facilitar análises temporais
    df.set_index("Timestamp", inplace=True)

    # Calcula o spread (diferença entre PAXG e XAUUSD)
    df["spread"] = df["close_y"] - df["close_x"]

    # Remove linhas com dados ausentes
    df.dropna(inplace=True)

    # Caminho do CSV processado
    output_csv = os.path.join(base_folder, "dados_com_spread.csv")
    df.to_csv(output_csv, index=True, date_format="%Y-%m-%d %H:%M:%S")

    logging.info(f"✅ CSV exportado com sucesso para: {output_csv}")
    logging.info(f"📊 Total de registros carregados: {df.shape[0]} linhas.")

    return df