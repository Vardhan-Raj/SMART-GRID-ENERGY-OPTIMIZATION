import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load forecasted demand
forecasted_df = pd.read_csv("data/processed/forecasted_energy.csv")
predicted_load = forecasted_df["predicted_load"].values

# Load PSO results
pso_df = pd.read_csv("data/processed/optimized_energy_distribution.csv")
pso_energy = pso_df["optimized_energy"].values

# Load ACO results
aco_df = pd.read_csv("data/processed/optimized_energy.csv")
aco_energy = aco_df["optimized_energy"].values

# Compute error metrics
def evaluate_performance(true_values, predicted_values, method):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    variance = np.var(predicted_values)
    
    print(f"📊 {method} Optimization Results:")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - Variance: {variance:.2f}")
    print("-" * 40)
    
    return mae, rmse, variance

# Evaluate PSO & ACO
pso_metrics = evaluate_performance(predicted_load, pso_energy, "PSO")
aco_metrics = evaluate_performance(predicted_load, aco_energy, "ACO")

# 📊 Visualization
plt.figure(figsize=(10, 5))
plt.plot(predicted_load, label="Forecasted Demand", color="black", linestyle="dashed")
plt.plot(pso_energy, label="PSO Optimized", color="blue")
plt.plot(aco_energy, label="ACO Optimized", color="red")
plt.xlabel("Grid ID")
plt.ylabel("Energy (kWh)")
plt.title("Comparison: PSO vs ACO vs Forecasted Demand")
plt.legend()
plt.grid(True)
plt.savefig("data/visualizations/pso_vs_aco_comparison.png")
plt.show()
