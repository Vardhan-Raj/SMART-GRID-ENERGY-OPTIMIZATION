import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import optuna

#  Enable XLA for faster training
tf.config.optimizer.set_jit(True)

#  Loading of dataset 
data_path = "data/processed/enriched_forecasted_energy.csv"
df = pd.read_csv(data_path)

#  Convert timestamp to datetime and extract time-based features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_holiday"] = df["timestamp"].dt.date.isin(
    pd.to_datetime(["2012-01-01", "2012-12-25"])  # Example holidays
).astype(int)

#  Feature Engineering: Select relevant columns for features and target
features = [
    "hour", "day_of_week", "is_holiday", "temperature", "humidity", "wind_speed", "precipitation",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24h", "rolling_max_24h", "rolling_min_24h"
]
target = "predicted_load"

#  Handle missing values, if the vallues ar missing
df.dropna(inplace=True)

#  Split data into train/test sets before scaling (80% train, 20% test)
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

#  Initialize scalers and scale data separately for train/test sets to avoid data leakage
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_df.loc[:, features] = feature_scaler.fit_transform(train_df[features])
train_df.loc[:, target] = target_scaler.fit_transform(train_df[[target]])
test_df.loc[:, features] = feature_scaler.transform(test_df[features])
test_df.loc[:, target] = target_scaler.transform(test_df[[target]])

#  Save the scalers for future use
joblib.dump(feature_scaler, "models/feature_scaler.pkl")
joblib.dump(target_scaler, "models/target_scaler.pkl")

#  Create sequences for LSTM model
def create_sequences(data, target_col, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # Features (exclude target column)
        y.append(data[i+seq_length, target_col])  # Target value at seq_length index
    return np.array(X), np.array(y)

SEQ_LENGTH = 24 # sets sequence length as 24 now

train_array = train_df[features + [target]].values
test_array = test_df[features + [target]].values

X_train, y_train = create_sequences(train_array, target_col=-1, seq_length=SEQ_LENGTH)
X_test, y_test = create_sequences(test_array, target_col=-1, seq_length=SEQ_LENGTH)

#  Convert data to tf.data.Dataset for efficient training
BATCH_SIZE = 64

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#  Define Optuna objective function for hyperparameter tuning
def objective(trial):
    try:
        lstm_units_1 = trial.suggest_int("lstm_units_1", 64, 128)
        lstm_units_2 = trial.suggest_int("lstm_units_2", 32, 64)
        dropout_rate_1 = trial.suggest_float("dropout_rate_1", 0.2, 0.5)
        dropout_rate_2 = trial.suggest_float("dropout_rate_2", 0.1, 0.3)

        model = Sequential([
            Input(shape=(SEQ_LENGTH, X_train.shape[2])),
            Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
            BatchNormalization(),
            Dropout(dropout_rate_1),
            Bidirectional(LSTM(lstm_units_2)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1)  # Output layer for regression task
        ])

        model.compile(optimizer=Adam(learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)), 
                      loss=Huber(), metrics=["mae"])

        history = model.fit(
            train_data,
            epochs=50,
            validation_data=test_data,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
            verbose=1
        )

        val_loss_min = min(history.history["val_loss"])
        return val_loss_min

    except Exception as e:
        print(f"Trial failed due to: {e}")
        return float("inf")

#  Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params

#  Build final optimized model using best parameters from Optuna study
model_final = Sequential([
    Input(shape=(SEQ_LENGTH, X_train.shape[2])),
    Bidirectional(LSTM(best_params["lstm_units_1"], return_sequences=True)),
    BatchNormalization(),
    Dropout(best_params["dropout_rate_1"]),
    Bidirectional(LSTM(best_params["lstm_units_2"])),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model_final.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss=Huber(), metrics=["mae"])

history_final = model_final.fit(
    train_data,
    epochs=50,
    validation_data=test_data,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
)

#  Save the trained model for future use
model_final.save("models/lstm_best_model_optuna.keras")

#  Evaluate the model on test data and compute metrics after inverse transformation
y_pred_scaled_final = model_final.predict(test_data)
y_test_inv_final = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv_final = target_scaler.inverse_transform(y_pred_scaled_final)

# calculating the regression metrics
mse_final = mean_squared_error(y_test_inv_final, y_pred_inv_final)
mae_final = mean_absolute_error(y_test_inv_final, y_pred_inv_final)
rmse_final = np.sqrt(mse_final)
r2_final = r2_score(y_test_inv_final, y_pred_inv_final)

#  Display evaluation/ regression metrics 
print("\nModel Evaluation Metrics:")
print(f" Mean Squared Error (MSE): {mse_final:.4f}")
print(f" Mean Absolute Error (MAE): {mae_final:.4f}")
print(f" Root Mean Squared Error (RMSE): {rmse_final:.4f}")
print(f" R² Score (Coefficient of Determination): {r2_final:.4f}")