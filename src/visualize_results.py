import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure directory exists
output_dir = "data/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Create figure for combined visualization
combined_fig = plt.figure(figsize=(12, 6))

# Function to load and plot data with error handling
def load_and_plot(file_path, color, label, position, width=0.25):
    try:
        df = pd.read_csv(file_path)
        
        # Plot on combined figure
        plt.figure(combined_fig.number)  # Return to combined figure
        plt.bar(np.array(df["grid_id"]) + position, 
                df["optimized_energy"], 
                width=width, 
                color=color, 
                label=label)
        
        # Create individual plot
        individual_fig = plt.figure(figsize=(10, 5))
        plt.bar(df["grid_id"], df["optimized_energy"], color=color)
        plt.xlabel("Grid ID")
        plt.ylabel("Optimized Energy (MW)")
        plt.title(f"{label} Energy Distribution Across Grids")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(output_dir, f"{label.lower()}_optimized_energy.png"))
        plt.close(individual_fig)
        return True
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping {label} visualization.")
        return False

# Load and plot each optimization result
has_pso = load_and_plot("data/processed/optimized_pso.csv", "skyblue", "PSO", -0.25)
has_aco = load_and_plot("data/processed/optimized_aco.csv", "lightcoral", "ACO", 0)

# Only add legends and save if at least one dataset was loaded
if any([has_pso, has_aco]):
    plt.figure(combined_fig.number)  # Return to combined figure
    plt.xlabel("Grid ID")
    plt.ylabel("Optimized Energy (MW)")
    plt.title("Comparison of Optimization Methods")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "optimization_comparison.png"))
    plt.show()
else:
    print("No optimization data files found. Please run the optimization algorithms first.")