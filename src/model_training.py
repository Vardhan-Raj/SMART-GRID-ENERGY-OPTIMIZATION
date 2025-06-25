import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

#  Enable XLA for faster training
tf.config.optimizer.set_jit(True)

#  Load dataset
data_path = "data/processed/updated_forecasted_energy.csv"
df = pd.read_csv(data_path)

#  Convert timestamp to datetime and extract time-based features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekend"] = (df["timestamp"].dt.weekday >= 5).astype(int)

#  Feature Engineering: Add rolling average and lagged features
df["rolling_mean"] = df["predicted_load"].rolling(window=24, min_periods=1).mean()
for lag in [1, 24]:  # Lagged features for 1 hour and 24 hours
    df[f"lag_{lag}"] = df["predicted_load"].shift(lag)
df.dropna(inplace=True)  # Drop rows with NaN values after adding lagged features

#  Select relevant columns for features and target
features = ["hour", "day", "month", "weekend", "rolling_mean", "lag_1", "lag_24"]
target = "predicted_load"

#  Split data into train/test sets before scaling to avoid data leakage (80% train, 20% test)
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

#  Initialize scalers and scale data separately for train/test sets
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_df[features] = feature_scaler.fit_transform(train_df[features])
train_df[target] = target_scaler.fit_transform(train_df[[target]])
test_df[features] = feature_scaler.transform(test_df[features])
test_df[target] = target_scaler.transform(test_df[[target]])

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

#  Prepare training and testing data as sequences
SEQ_LENGTH = 24  # Reduced sequence length to match daily patterns

train_array = train_df[features + [target]].values
test_array = test_df[features + [target]].values

X_train, y_train = create_sequences(train_array, target_col=-1, seq_length=SEQ_LENGTH)
X_test, y_test = create_sequences(test_array, target_col=-1, seq_length=SEQ_LENGTH)

#  Convert data to tf.data.Dataset for efficient training
BATCH_SIZE = 64 

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#  Define a learning rate scheduler to dynamically adjust learning rate during training
lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

#  Define an optimized LSTM model architecture
model = Sequential([
    Input(shape=(SEQ_LENGTH, X_train.shape[2])),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)  # Output layer for regression task
])

#  Compile the model with Adam optimizer and Huber loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=["mae"])

#  Early stopping to prevent overfitting and restore best weights
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#  Train the model with validation on test data
history = model.fit(
    train_data,
    epochs=100,
    validation_data=test_data,
    callbacks=[early_stop, lr_schedule],
    verbose=1,
)

model.save("models/lstm_best_model.keras")

#  Evaluate the model on test data and compute metrics after inverse transformation
y_pred_scaled = model.predict(test_data)
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = target_scaler.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n📊 Model Evaluation Metrics:")
print(f" Mean Squared Error (MSE): {mse:.4f}")
print(f" Mean Absolute Error (MAE): {mae:.4f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" R² Score (Coefficient of Determination): {r2:.4f}")