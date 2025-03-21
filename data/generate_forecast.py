import pandas as pd
import numpy as np
import os

# Ensure directory exists
os.makedirs("data/processed", exist_ok=True)

# Generate timestamps for 30 days (hourly data)
date_rng = pd.date_range(start="2024-02-01", periods=720, freq="H")

# Simulated LSTM predictions (random walk with trend)
predicted_load = np.cumsum(np.random.normal(loc=0.3, scale=2, size=len(date_rng))) + 450

# Create DataFrame
df = pd.DataFrame({"timestamp": date_rng, "predicted_load": predicted_load})

# Save CSV
df.to_csv("data/processed/forecasted_energy.csv", index=False)

print("✅ Forecasted energy data saved at 'data/processed/forecasted_energy.csv'")
print(df.head())  # Show first few rows
