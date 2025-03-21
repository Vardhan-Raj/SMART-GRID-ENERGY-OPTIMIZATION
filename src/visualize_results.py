import os
import pandas as pd
import matplotlib.pyplot as plt

# Load optimization results
df = pd.read_csv("data/processed/optimized_energy_distribution.csv")

# Ensure directory exists
output_dir = "data/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(df["grid_id"], df["optimized_energy"], color="skyblue")
plt.xlabel("Grid ID")
plt.ylabel("Optimized Energy (MW)")
plt.title("Optimized Energy Distribution Across Grids")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.savefig(os.path.join(output_dir, "optimized_energy.png"))
plt.show()