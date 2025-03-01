import os
import joblib
import numpy as np
import pandas as pd
import logging

# Configuração do logging para a aba de testes
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Caminhos dos modelos treinados
models_folder = r"C:\Users\samue\Documents\Projeto_Quant_H2\models"
buy_model_path = os.path.join(models_folder, "buy_model.pkl")
sell_model_path = os.path.join(models_folder, "sell_model.pkl")

# 🚀 **1️⃣ Carregar os modelos treinados**
logging.info("🔍 Carregando modelos...")
try:
    buy_model = joblib.load(buy_model_path)
    sell_model = joblib.load(sell_model_path)
    logging.info("✅ Modelos carregados com sucesso!")
except Exception as e:
    logging.error(f"❌ Erro ao carregar modelos: {e}")
    raise e

# 🚀 **2️⃣ Verificar as features esperadas pelo modelo**
expected_features = buy_model.get_booster().feature_names
logging.info(f"📌 Features esperadas pelo modelo: {expected_features}")

# 🚀 **3️⃣ Criar um DataFrame de teste com features simuladas**
num_samples = 100  # Número de amostras para teste
df_test = pd.DataFrame(np.random.rand(num_samples, len(expected_features)), columns=expected_features)

# 🚀 **4️⃣ Garantir que as features estão na mesma ordem e tipo**
df_test = df_test.astype(np.float32)

# 🚀 **5️⃣ Testar a predição dos modelos**
try:
    logging.info("🔍 Testando predição do modelo...")
    buy_probs = buy_model.predict_proba(df_test)[:, 1]
    sell_probs = sell_model.predict_proba(df_test)[:, 1]

    # Verifica se os valores preditos estão dentro do intervalo [0,1]
    if np.any((buy_probs < 0) | (buy_probs > 1)) or np.any((sell_probs < 0) | (sell_probs > 1)):
        raise ValueError("❌ ERRO: As probabilidades preditas não estão no intervalo [0,1]!")

    logging.info(f"✅ Teste de predição bem-sucedido! Exemplo de probabilidades:")
    logging.info(f"📈 Compra: {buy_probs[:5]}")
    logging.info(f"📉 Venda: {sell_probs[:5]}")

except Exception as e:
    logging.error(f"❌ ERRO ao testar o modelo: {e}")
    raise e

logging.info("✅ Modelos validados com sucesso! Tudo pronto para rodar o pipeline.")
