import os
import numpy as np
import matplotlib.pyplot as plt
from engibench.problems.thermoelastic3d import ThermoElastic3D

def run_3d_experiments_safe():
    output_dir = "safe_results_3d"
    os.makedirs(output_dir, exist_ok=True)

    # Reverting to the native, safe dimensions for ThermoElastic3D
    nx=ny=nz=32

    configs = [
        {"max_iter": 30, "volfrac": 0.2, "weight": 1.0, "name": "3D_32x32x32_light"},
        {"max_iter": 30, "volfrac": 0.5, "weight": 0.5, "name": "3D_32x32x32_heavy"}
    ]

    problem = ThermoElastic3D(seed=42)

    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n--- Starting 3D Experiment: {name} ---")
        print(f"Grid Size: {nx}x{ny}x{nz} ({nx * ny * nz} total elements)")

        # 1. SETUP STARTING POINT (Matching the 16x16x16 constraint)
        starting_design = cfg["volfrac"] * np.ones((nx, ny, nz), dtype=float)

        # 2. OPTIMIZE
        print("3D Optimization in progress...")
        optimal_design, history = problem.optimize(starting_point=starting_design, config=cfg)

        # 3. SIMULATE
        results = problem.simulate(optimal_design, config=cfg)
        compliance_structural = results[0]
        compliance_thermal = results[1]
        print(f"Results: Structural Comp. = {compliance_structural:.4f} | Thermal Comp. = {compliance_thermal:.4f}")

        # 4. RENDER
        print("Generating 3D render...")
        rendered_pixels = problem.render(optimal_design, open_window=True)

        design_path = os.path.join(output_dir, f"design_{name}.png")
        plt.imsave(design_path, rendered_pixels)
        print(f"Saved 3D render to: {design_path}")

        # 5. PERFORMANCE PLOT
        str_compliance_history = [step.obj_values[0] for step in history]

        plt.figure(figsize=(8, 4))
        plt.plot(range(len(str_compliance_history)), str_compliance_history, marker='o', markersize=3, color='red')
        plt.title(f"3D Structural Compliance Convergence ({name})")
        plt.xlabel("Iteration")
        plt.ylabel("Structural Compliance")
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"convergence_{name}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved performance plot to: {plot_path}")

if __name__ == "__main__":
    run_3d_experiments_safe()
