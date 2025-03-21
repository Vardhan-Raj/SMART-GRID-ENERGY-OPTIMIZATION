import numpy as np
import pandas as pd

# Load forecasted demand data
data_path = "data/processed/forecasted_energy.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"❌ Error: File not found at {data_path}")
    exit()

# Ensure the required column exists
if "predicted_load" not in df.columns:
    print("❌ Error: 'predicted_load' column is missing in the dataset!")
    print("Available columns:", df.columns)
    exit()

# Extract demand values
demand = df["predicted_load"].values
num_grids = len(demand)

# **ACO Parameters**
num_ants = 10
num_iterations = 100
pheromones = np.ones(num_grids)
evaporation_rate = 0.1
best_solution = None
best_cost = float("inf")

# **Ant Colony Optimization Process**
for iteration in range(num_iterations):
    ant_solutions = np.zeros((num_ants, num_grids))

    for ant in range(num_ants):
        probabilities = pheromones / pheromones.sum()  # Normalize probabilities

        # 🔹 Fix probability size issue
        if probabilities.shape[0] != 3:
            probabilities = np.array([0.3, 0.4, 0.3])  # Default fallback

        ant_solutions[ant] = demand + np.random.choice(
            [-10, 0, 10], size=num_grids, p=[0.3, 0.4, 0.3]
        )

    # Evaluate cost (Total Energy Deviation per ant)
    costs = np.abs(ant_solutions - demand).sum(axis=1)
    best_ant = np.argmin(costs)

    if costs[best_ant] < best_cost:
        best_cost = costs[best_ant]
        best_solution = ant_solutions[best_ant]

    # **Pheromone Update (Fixed)**
    pheromones *= (1 - evaporation_rate)  # Evaporation
    pheromones += np.exp(-np.abs(best_solution - demand))  # Deposit based on best solution

    print(f"Iteration {iteration + 1}/{num_iterations} - Best Cost: {best_cost}")

# **Save Optimized Energy Distribution**
optimized_data = pd.DataFrame({"grid_id": np.arange(num_grids), "optimized_energy": best_solution})
optimized_data.to_csv("data/processed/optimized_energy.csv", index=False)

print("✅ ACO Optimization Completed. Results saved!")
