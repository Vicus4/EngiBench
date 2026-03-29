import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import Gabriel's ThermoElastic3D class
from engibench.problems.thermoelastic3d.v0 import ThermoElastic3D

def map_forcedist_to_matrix(forcedist: float, res: int):
    """Maps a float value [0, 1) to a specific node on the top XY plane (Z max)."""
    nodes_per_side = res + 1
    total_nodes = nodes_per_side * nodes_per_side

    node_idx = int(forcedist * total_nodes)
    if node_idx >= total_nodes:
        node_idx = total_nodes - 1

    pos_x = node_idx // nodes_per_side
    pos_y = node_idx % nodes_per_side

    force_matrix = np.zeros((nodes_per_side, nodes_per_side, nodes_per_side), dtype=int)
    force_matrix[pos_x, pos_y, -1] = 1 # Force applied on the top face

    return force_matrix

def get_fixed_elements(res: int):
    """Generates the boundary condition matrix based on resolution (4 bottom corners fixed)."""
    fixed = np.zeros((res + 1, res + 1, res + 1), dtype=int)
    fixed[0, 0, 0] = 1
    fixed[-1, 0, 0] = 1
    fixed[0, -1, 0] = 1
    fixed[-1, -1, 0] = 1
    return fixed

def generate_single_sample(args):
    """Runs a single optimization using Gabriel's ThermoElastic3D as a pure structural problem."""
    res, vol_frac, r_min, force_dist, seed = args

    problem = ThermoElastic3D(seed=seed)

    # Generate forces and boundary conditions for the correct resolution
    force_z_matrix = map_forcedist_to_matrix(force_dist, res)
    fixed_matrix = get_fixed_elements(res)
    empty_matrix = np.zeros((res + 1, res + 1, res + 1), dtype=int)

    # Configure Gabriel's model to act purely structurally (weight = 1.0)
    config = {
        "nelx": res,
        "nely": res,
        "nelz": res,
        "volfrac": vol_frac,
        "rmin": r_min,
        "weight": 1.0,  # CRITICAL: 1.0 = Pure Structural, 0.0 = Pure Thermal
        "force_elements_z": force_z_matrix,
        "fixed_elements": fixed_matrix,
        "force_elements_x": empty_matrix,  # No lateral forces
        "force_elements_y": empty_matrix,  # No lateral forces
        "heatsink_elements": empty_matrix, # No heatsinks
        "max_iter": 100
    }

    starting_point = vol_frac * np.ones((res, res, res), dtype=np.float32)

    try:
        optimized_design, opti_steps = problem.optimize(starting_point, config=config)

        # Evaluate the final design using simulate.
        # ThermoElastic3D simulate returns: [structural_compliance, thermal_compliance, volume_fraction]
        final_metrics = problem.simulate(optimized_design, config=config)

        return {
            "design": optimized_design,
            "volfrac": vol_frac,
            "rmin": r_min,
            "forcedist": force_dist,
            # Extract only the structural compliance (index 0)
            "final_compliance": float(final_metrics[0])
        }
    except Exception as e:
        return {"error": str(e), "res": res}

def main():
    # We are generating 5 specific samples to match your initial run
    samples_per_res = 5
    resolutions = [16]

    # Note: We are using a fixed set of specific parameters here to ensure it uses the exact same input as your Beams3D CSV.
    # We will pass these exact parameters to the task list.
    target_samples = [
        {"volfrac": 0.4229358633246718, "rmin": 2.0, "forcedist": 0.5248774407330495},
        {"volfrac": 0.2062, "rmin": 2.0, "forcedist": 0.3301}, # Approximated from CSV
        {"volfrac": 0.4359, "rmin": 1.0, "forcedist": 0.1345}, # Approximated from CSV
        {"volfrac": 0.3096, "rmin": 1.0, "forcedist": 0.3721}, # Approximated from CSV
        {"volfrac": 0.2948, "rmin": 1.0, "forcedist": 0.6504}  # Approximated from CSV
    ]

    # Directory to save the output (same as Beams3D)
    # Adjust this path if your dataset_output is located elsewhere relative to this script
    save_dir = "../../../dataset_output"
    os.makedirs(save_dir, exist_ok=True)

    # Get CPU cores for multiprocessing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting ThermoElastic comparison dataset generation using {num_cores} CPU cores.")

    for res in resolutions:
        print(f"\n--- Generating dataset for resolution: {res}x{res}x{res} ---")

        # Prepare tasks using the exact 5 samples from your previous run
        tasks = []
        for i, sample_data in enumerate(target_samples):
            seed = 12345 + i # Deterministic seed for reproducibility
            tasks.append((res, sample_data["volfrac"], sample_data["rmin"], sample_data["forcedist"], seed))

        res_data = []
        successful_samples = 0

        # Execute tasks in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(generate_single_sample, task): task for task in tasks}

            for future in tqdm(as_completed(futures), total=samples_per_res, desc=f"Res {res}"):
                result = future.result()

                if "error" not in result:
                    res_data.append(result)
                    successful_samples += 1
                else:
                    print(f"\nError generating sample at res {res}: {result['error']}")

        # Final save with a specific "thermoelastic" prefix
        print(f"\nSaving final data for resolution {res}...")
        save_path = os.path.join(save_dir, f"thermoelastic_resolution_{res}_final.npz")
        np.savez_compressed(save_path, data=res_data)
        print(f"Resolution {res} completed. Saved {successful_samples}/{samples_per_res} valid samples to {save_path}.")

if __name__ == "__main__":
    main()
