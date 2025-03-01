import os
import logging
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_logs.log"), logging.StreamHandler()]
)

# 1️⃣ **Criação de Target Dinâmico**
def create_dynamic_target(df, min_horizon, max_horizon, percentile_factor, price_threshold, target_name, condition_function):
    """
    Creates a dynamic target variable based on specified condition.
    """
    df = df.copy()
    
    # Calculate adaptive forecast horizon based on volatility, handling NA values safely
    volatility = df["spread"].rolling(window=24*30).std()
    
    # Substituir valores NaN ou infinitos na volatilidade por um valor padrão
    volatility = volatility.replace([np.inf, -np.inf], np.nan)
    volatility = volatility.fillna(volatility.median() if not np.isnan(volatility.median()) else min_horizon/percentile_factor)
    
    # Calcular horizonte de forma segura para evitar overflow
    raw_horizon = (volatility * percentile_factor).clip(0, 10000)  # Limitar para evitar valores extremos
    
    # Converter para inteiro de forma segura
    horizon_int = np.floor(raw_horizon).astype('Int64')  # Usar Int64 que suporta NA
    
    # Aplicar limites min e max
    df["forecast_horizon"] = np.maximum(min_horizon, 
                             np.minimum(max_horizon, horizon_int)).fillna(min_horizon)
    
    # Calcular future_spread para cada linha usando seu próprio horizonte
    future_values = np.full(len(df), np.nan)
    
    # Método eficiente usando numpy e iloc
    for i in range(len(df)):
        horizon = int(df["forecast_horizon"].iloc[i])
        if i + horizon < len(df):
            future_values[i] = df["spread"].iloc[i + horizon]
    
    df["future_spread"] = future_values
    
    # Calculate price change percentage
    df["price_change"] = (df["future_spread"] - df["spread"]) / df["spread"]
    
    # Create target based on condition
    df[target_name] = 0
    mask = condition_function(df) & (df["price_change"].abs() > price_threshold)
    df.loc[mask, target_name] = 1
    
    # Drop rows where target cannot be calculated (at the end of the series)
    df.dropna(subset=["future_spread"], inplace=True)
    
    logging.info(f"✅ Target '{target_name}' criado com sucesso. Distribuição: {df[target_name].mean()*100:.2f}% positivos")
    
    return df

def create_buy_target(df):
    """Cria target para COMPRA (spread futuro > spread atual)."""
    df = df.copy()
    return create_dynamic_target(df, 60, 24, 90, 0.02, "buy_target", lambda df: df["future_spread"] > df["spread"])

def create_sell_target(df):
    """Cria target para VENDA (spread futuro < spread atual)."""
    df = df.copy()
    return create_dynamic_target(df, 48, 12, 72, 0.015, "sell_target", lambda df: df["future_spread"] < df["spread"])

# 2️⃣ **Treinamento do Modelo**
def check_features(df, features):
    """Verifica se as features estão presentes no DataFrame e preenche valores ausentes."""
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        logging.warning(f"⚠️ Features ausentes: {missing_features} ({len(missing_features)} no total)")
        for feature in missing_features:
            df[feature] = 0.0  # Adiciona coluna com valores neutros

    logging.info(f"✅ Features confirmadas: {features}")
    
    return features

def train_xgboost(df, features, target_col="target", optimize_hyperparams=False, model_path=None, custom_params=None):
    """Treina um modelo XGBoost e salva o modelo treinado."""
    logging.info(f"Treinando modelo para {target_col}. Otimização: {optimize_hyperparams}")

    if df.empty or target_col not in df.columns:
        raise ValueError(f"❌ DataFrame inválido para treinamento. Target {target_col} ausente!")

    features = check_features(df, features)
    df = df.dropna(subset=[target_col])

    X, y = df[features], df[target_col]

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if y_train.nunique() < 2:
        raise ValueError("❌ Target tem apenas uma classe no conjunto de treino. Impossível treinar o modelo!")

    tscv = TimeSeriesSplit(n_splits=5)

    # Balanceamento de classes
    if y_train.value_counts(normalize=True).min() < 0.2:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    model_params = {
        "objective": "binary:logistic",
        "learning_rate": 0.005,
        "max_depth": 15,
        "n_estimators": 2000,
        "subsample": 1,
        "colsample_bytree": 1,
        "gamma": 0.0,
        "min_child_weight": 1,
        "random_state": 42
    }

    if custom_params:
        model_params.update(custom_params)

    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0)
    }

    logging.info(f"📌 Modelo treinado com métricas: {metrics}")

    if model_path:
        joblib.dump(model, model_path)
        logging.info(f"💾 Modelo salvo em: {model_path}")

    return model, metrics

# 3️⃣ **Treinar Modelos Duais**
def train_dual_models(df, buy_features, sell_features, optimize_hyperparams=False, model_folder="models"):
    """Treina modelos separados para compra e venda e salva os modelos."""
    os.makedirs(model_folder, exist_ok=True)

    buy_model_path = os.path.join(model_folder, "buy_model.pkl")
    sell_model_path = os.path.join(model_folder, "sell_model.pkl")

    logging.info("🛠️ Treinando novos modelos XGBoost...")

    df_buy = create_buy_target(df)
    df_sell = create_sell_target(df)

    buy_model, _ = train_xgboost(df_buy, buy_features, "buy_target", optimize_hyperparams, buy_model_path)
    sell_model, _ = train_xgboost(df_sell, sell_features, "sell_target", optimize_hyperparams, sell_model_path)

    return buy_model, sell_model

def generate_dual_signals(df, buy_model, sell_model, buy_threshold=0.7, sell_threshold=0.7):
    """
    Gera sinais de compra/venda usando os modelos treinados, 
    validando e ajustando as features para manter consistência.
    """

    df = df.copy()  # Evita modificar o DataFrame original

    # 🔹 Recupera as features usadas no treinamento do modelo
    expected_buy_features = buy_model.get_booster().feature_names if buy_model else []
    expected_sell_features = sell_model.get_booster().feature_names if sell_model else []
    if not expected_buy_features or not expected_sell_features:
        raise ValueError("❌ ERRO: Os modelos não contêm features treinadas!")

    logging.info(f"🔍 Features esperadas no modelo de compra: {expected_buy_features}")
    logging.info(f"🔍 Features esperadas no modelo de venda: {expected_sell_features}")

    # 🔹 Ajusta "atr" se necessário
    if "atr_x" in expected_buy_features and "atr_x" in df.columns:
        df["atr_x"] = df["atr_x"]
    else:
        df["atr_x"] = 0.0

    if "atr_y" in expected_sell_features and "atr_y" in df.columns:
        df["atr_y"] = df["atr_y"]
    else:
        df["atr_y"] = 0.0

    # 🔹 Cria qualquer feature ausente
    missing_buy_features = [f for f in expected_buy_features if f not in df.columns]
    if missing_buy_features:
        logging.warning(f"⚠️ Features de compra ausentes! Criando com valor 0.0: {missing_buy_features}")
        for feature in missing_buy_features:
            df[feature] = 0.0

    missing_sell_features = [f for f in expected_sell_features if f not in df.columns]
    if missing_sell_features:
        logging.warning(f"⚠️ Features de venda ausentes! Criando com valor 0.0: {missing_sell_features}")
        for feature in missing_sell_features:
            df[feature] = 0.0

    # 🔹 Remove features extras
    extra_features = [f for f in df.columns if f not in expected_buy_features and f not in expected_sell_features]
    if extra_features:
        logging.warning(f"⚠️ Removendo features extras: {extra_features}")
        df.drop(columns=extra_features, inplace=True)

    # 🔹 Organiza as colunas na ordem esperada
    df_buy = df[expected_buy_features]
    df_sell = df[expected_sell_features]
    df_buy.fillna(0, inplace=True)  # Evita falhas por NaNs
    df_sell.fillna(0, inplace=True)  # Evita falhas por NaNs
    logging.info("✅ Valores NaN substituídos por 0 antes da predição.")

    # 🔹 Converte para float32 e gera probabilidades
    X_buy = df_buy.astype(np.float32)
    X_sell = df_sell.astype(np.float32)
    try:
        # Use .to_numpy() para garantir que não haja verificação de nomes de features
        df["buy_prob"] = buy_model.predict_proba(X_buy.to_numpy())[:, 1]
        df["sell_prob"] = sell_model.predict_proba(X_sell.to_numpy())[:, 1]
    except Exception as e:
        logging.error(f"❌ ERRO ao prever probabilidades: {e}")
        raise e

    # 🔹 Gera sinais com base nos thresholds
    if "buy_prob" in df.columns and "sell_prob" in df.columns:
        df["buy_signal"] = np.where(df["buy_prob"] > buy_threshold, 1, 0)
        df["sell_signal"] = np.where(df["sell_prob"] > sell_threshold, -1, 0)
        df["signal"] = np.where(df["sell_signal"] != 0, df["sell_signal"], df["buy_signal"])
    else:
        df["signal"] = 0
        logging.error("❌ ERRO: Não foi possível gerar sinais! Inicializando com 0.")

    # 🔍 Verificação final
    if "signal" not in df.columns:
        df["signal"] = 0
        logging.error("❌ ERRO: 'signal' não foi gerada! Inicializando com 0.")

    logging.info(f"✅ Sinais gerados: {df['signal'].sum()} (positivos=compra, negativos=venda)")
    return df