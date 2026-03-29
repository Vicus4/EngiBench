import numpy as np
import csv

file_path = "resolution_16_final.npz"
csv_filename = "dataset_results_res16.csv"

res = 16
nodes_per_side = res + 1
total_nodes = nodes_per_side * nodes_per_side

try:
    loaded_file = np.load(file_path, allow_pickle=True)
    dataset = loaded_file['data']

    print(f"\n✅ FOUND {len(dataset)} SAMPLES IN THE DATASET.")
    print(f"Saving data to '{csv_filename}'...")
    print("-" * 50)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Intestazioni delle colonne (ho aggiunto Norm_X e Norm_Y)
        writer.writerow([
            "Sample_ID", "Volume_Fraction", "R_min", "Compliance",
            "Force_Dist_0_to_1", "Force_X", "Force_Y", "Force_Z",
            "Norm_X_0_to_1", "Norm_Y_0_to_1"
        ])

        for i, sample in enumerate(dataset):
            # Estraiamo come float puri per sicurezza
            volfrac = float(sample['volfrac'])
            rmin = float(sample['rmin'])
            compliance = float(sample['final_compliance'])

            forcedist = sample['forcedist']
            if isinstance(forcedist, np.ndarray) and forcedist.ndim == 0:
                forcedist = float(forcedist.item())
            else:
                forcedist = float(forcedist)

            # Calcoliamo le coordinate assolute
            node_idx = int(forcedist * total_nodes)
            if node_idx >= total_nodes:
                node_idx = total_nodes - 1

            pos_x = node_idx // nodes_per_side
            pos_y = node_idx % nodes_per_side
            pos_z = res

            # Calcoliamo le coordinate normalizzate su scala 0-1
            norm_x = pos_x / res
            norm_y = pos_y / res

            # Salviamo nel CSV arrotondando a 4 decimali per una lettura perfetta
            writer.writerow([
                i,
                f"{volfrac:.4f}",
                f"{rmin:.1f}",
                f"{compliance:.4f}",
                f"{forcedist:.4f}",
                pos_x, pos_y, pos_z,
                f"{norm_x:.4f}", f"{norm_y:.4f}"
            ])

            print(f"Sample {i} -> Force_Dist: {forcedist:.4f} | Coord_Norm: ({norm_x:.2f}, {norm_y:.2f})")

    print("-" * 50)
    print(f"🎉 DONE! The CSV file is formatted perfectly.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
