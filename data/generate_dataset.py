import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for 30 days, hourly data
date_rng = pd.date_range(start="2024-01-01", end="2024-01-30", freq="H")

# Simulated energy consumption (random walk with trend)
energy_consumption = np.cumsum(np.random.normal(loc=0.5, scale=5, size=len(date_rng))) + 450

# Create DataFrame
df = pd.DataFrame({"timestamp": date_rng, "energy_consumption": energy_consumption})

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)

# Save as CSV
df.to_csv("data/raw/electricity_consumption.csv", index=False)

print("✅ Dataset generated successfully! Saved at 'data/raw/electricity_consumption.csv'")
print(df.head())  # Show first few rows
