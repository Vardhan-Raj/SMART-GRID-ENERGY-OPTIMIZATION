import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("tuner_results", exist_ok=True)

# Load dataset
data_path = "data/processed/forecasted_energy.csv"
data = pd.read_csv(data_path)

# Data Preprocessing
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.iloc[:, 1:])  # Normalize all except time column

# Convert to supervised learning format
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, 0])  # Forecast demand
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(data_scaled, time_steps)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Hypermodel
def build_lstm_model(hp):
    model = keras.Sequential()
    
    # Tune the number of LSTM layers
    for i in range(hp.Int("num_layers", 1, 3)):  
        model.add(layers.LSTM(units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
                              return_sequences=True if i < hp.Int("num_layers", 1, 3) - 1 else False))
        model.add(layers.Dropout(hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(layers.Dense(1, activation="linear"))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="mse",
        metrics=["mae"]
    )
    
    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    build_lstm_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=2,
    directory="tuner_results",
    project_name="lstm_tuning"
)

# Start tuning
tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")

# Train final model with best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)

# Save model
best_model.save("models/lstm_best_model.h5")
print("✅ Best LSTM model saved at models/lstm_best_model.h5")
