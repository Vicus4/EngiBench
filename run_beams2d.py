import os
import matplotlib.pyplot as plt
from engibench.problems.beams2d import Beams2D

def run_experiments():
    # Create the output directory. If it already exists, no error is raised.
    output_dir = "first_results"
    os.makedirs(output_dir, exist_ok=True)

    # Define configurations: one lightweight and one more massive
    configs = [
        {"nelx": 60, "nely": 30, "volfrac": 0.2, "rmin": 1.0, "max_iter": 60},
        {"nelx": 60, "nely": 30, "volfrac": 0.5, "rmin": 2.0, "max_iter": 60}
    ]
    names = ["lightweight_structure", "massive_structure"]

    for cfg, name in zip(configs, names):
        print(f"\n--- Starting Experiment: {name} ---")

        # 1. SETUP: Initialize the Beams2D class with the chosen parameters
        problem = Beams2D(seed=42, config=cfg)

        # 2. OPTIMIZE: Start the mathematical shape-finding cycle
        print("Optimization in progress (calculating design)...")
        optimal_design, history = problem.optimize()

        # 3. SIMULATE: Physical verification of the final performance
        compliance = problem.simulate(optimal_design)[0]
        print(f"Final compliance ({name}): {compliance:.4f}")

        # 4. RENDER: Visualize the material distribution
        fig, ax = problem.render(optimal_design)
        design_path = os.path.join(output_dir, f"design_{name}.png")
        fig.savefig(design_path, bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved design render to: {design_path}")

        # 5. CONVERGENCE PLOT: Extract historical data
        obj_values = [step.obj_values[0] for step in history]

        plt.figure(figsize=(8, 4))
        plt.plot(range(len(obj_values)), obj_values, marker='o', markersize=3, linestyle='-', color='b')
        plt.title(f"Compliance Convergence ({name})")
        plt.xlabel("Iteration")
        plt.ylabel("Compliance (Lower is better)")
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"convergence_{name}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved performance plot to: {plot_path}")

if __name__ == "__main__":
    run_experiments()
