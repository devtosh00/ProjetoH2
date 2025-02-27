import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, n_steps, horizon=1):
    X, y = [], []
    for i in range(n_steps, len(data) - horizon + 1):
        X.append(data[i - n_steps:i, 0])
        y.append(data[i + horizon - 1, 0])
    return np.array(X), np.array(y)

def train_lstm_model(spread_series, n_steps=30, forecast_horizon=5, epochs=50, batch_size=32):
    # Escalonar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    spread_data = spread_series.values.reshape(-1, 1)
    spread_scaled = scaler.fit_transform(spread_data)
    
    # Criar sequências
    X, y = create_sequences(spread_scaled, n_steps, forecast_horizon)
    X = X.reshape((X.shape[0], n_steps, 1))
    
    # Dividir em treino/teste (70/30)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Construir o modelo LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.1),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    
    # Previsão para todo o dataset (opcional)
    X_all, _ = create_sequences(spread_scaled, n_steps, forecast_horizon)
    X_all = X_all.reshape(-1, n_steps, 1)
    predictions = model.predict(X_all)
    predictions_inv = scaler.inverse_transform(predictions)
    
    return model, predictions_inv, scaler
