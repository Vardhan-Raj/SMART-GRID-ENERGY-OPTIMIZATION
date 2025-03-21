import numpy as np
import pandas as pd
from pyswarm import pso

# Load forecasted energy demand
def load_forecasted_data():
    file_path = "data/processed/forecasted_energy.csv"
    df = pd.read_csv(file_path)
    return df["predicted_load"].values  # Ensure correct column name

# Define the cost function for PSO
def cost_function(energy_distribution, *args):
    demand = args[0]  # Actual forecasted demand

    # Total difference between allocated energy and demand
    total_imbalance = np.abs(np.sum(energy_distribution) - np.sum(demand))

    # Penalizing grids that receive too much or too little energy
    distribution_penalty = np.sum(np.abs(energy_distribution - demand[:len(energy_distribution)]))

    # Penalizing variance to encourage fair distribution
    variance_penalty = 5 * np.std(energy_distribution)

    # Total cost function
    return total_imbalance + variance_penalty + distribution_penalty

# Optimize energy distribution using PSO
def optimize_energy_distribution():
    demand = load_forecasted_data()
    num_grids = len(demand)

    # Lower and upper bounds for energy distribution
    lb = [50] * num_grids  # Minimum energy allocation
    ub = [500] * num_grids  # Maximum energy allocation

    # Run PSO
    optimized_distribution, _ = pso(cost_function, lb, ub, args=(demand,), maxiter=100, swarmsize=30)

    # Save results
    df = pd.DataFrame({
        "grid_id": range(1, num_grids + 1),
        "optimized_energy": optimized_distribution
    })
    df.to_csv("data/processed/optimized_energy_distribution.csv", index=False)

    print("\n✅ PSO Optimization Completed. Results saved at 'data/processed/optimized_energy_distribution.csv'.")
    print(df.head())

# Run optimization
if __name__ == "__main__":
    optimize_energy_distribution()
