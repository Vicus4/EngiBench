import numpy as np

# Replace 'NAME_OF_YOUR_FILE.npz' with the exact path of the file you want to check
# for example: 'dataset_output/res_16/sample_0.npz'
file_path = "resolution_16_final.npz" 

try:
    data = np.load(file_path, allow_pickle=True)
    
    print("\n--- FILE CONTENT ---")
    bcs = data['bcs'].item()
    compliance = data['compliance']
    
    print(f"Volume Fraction: {bcs['volfrac']}")
    print(f"Structural Compliance (Beams3D): {compliance:.4f}")
    
    # Let's find exactly where the force was applied
    force_matrix = bcs['force_elements_z']
    x_idx, y_idx, z_idx = np.where(force_matrix == 1)
    
    print(f"Force position -> X: {x_idx[0]}, Y: {y_idx[0]}")
    print("--------------------\n")
    
except Exception as e:
    print(f"Error opening the file: {e}")