import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Importiamo la classe dal file v0.py che si trova nella stessa cartella
from engibench.problems.beams3d.v0 import Beams3D

def map_forcedist_to_matrix(forcedist: float, res: int):
    """Mappa un valore float [0, 1) a un nodo specifico sul piano superiore XY."""
    nodes_per_side = res + 1
    total_nodes = nodes_per_side * nodes_per_side
    
    node_idx = int(forcedist * total_nodes)
    if node_idx >= total_nodes: 
        node_idx = total_nodes - 1
        
    pos_x = node_idx // nodes_per_side
    pos_y = node_idx % nodes_per_side
    
    force_matrix = np.zeros((nodes_per_side, nodes_per_side, nodes_per_side), dtype=int)
    force_matrix[pos_x, pos_y, -1] = 1 # Forza applicata sulla faccia superiore (Z max)
    
    return force_matrix
def get_fixed_elements(res: int):
    """Genera la matrice dei vincoli in base alla risoluzione."""
    fixed = np.zeros((res + 1, res + 1, res + 1), dtype=int)
    # Blocchiamo i 4 angoli inferiori (z=0) come nel problema di default
    fixed[0, 0, 0] = 1
    fixed[-1, 0, 0] = 1
    fixed[0, -1, 0] = 1
    fixed[-1, -1, 0] = 1
    return fixed

def generate_single_sample(args):
    res, vol_frac, r_min, force_dist, seed = args
    
    problem = Beams3D(seed=seed)
    
    # Generiamo sia le forze che i vincoli per la risoluzione corretta
    force_z_matrix = map_forcedist_to_matrix(force_dist, res)
    fixed_matrix = get_fixed_elements(res)

    config = {
        "nelx": res,
        "nely": res,
        "nelz": res,
        "volfrac": vol_frac,
        "rmin": r_min,
        "force_elements_z": force_z_matrix,
        "fixed_elements": fixed_matrix,  # <-- AGGIUNTO QUESTO
        "max_iter": 100 
    }

    starting_point = vol_frac * np.ones((res, res, res), dtype=np.float32)

    try:
        optimized_design, opti_steps = problem.optimize(starting_point, config=config)
        
        # Valutiamo il design finale chiamando simulate, che ci restituisce un array:
        # [structural_compliance, volume_fraction]
        final_metrics = problem.simulate(optimized_design, config=config)
        
        return {
            "design": optimized_design,
            "volfrac": vol_frac,
            "rmin": r_min,
            "forcedist": force_dist,
            "final_compliance": float(final_metrics[0])
        }
    except Exception as e:
        return {"error": str(e), "res": res}

def main():
    samples_per_res = 5 #500
    resolutions = [16] #, 32, 64]
    rmin_options = [1.0, 2.0, 3.0, 4.0]
    
    # Cartella dove salveremo i dati (verrà creata dentro beams3d)
    save_dir = "dataset_output"
    os.makedirs(save_dir, exist_ok=True)

    # Otteniamo il numero di core della CPU (lasciamone uno libero per non bloccare il PC)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Avvio generazione dataset utilizzando {num_cores} core della CPU.")

    for res in resolutions:
        print(f"\n--- Generazione dataset per risoluzione: {res}x{res}x{res} ---")
        
        # Prepariamo tutti i task (le combinazioni randomiche) per questa risoluzione
        tasks = []
        for i in range(samples_per_res):
            vol_frac = np.random.uniform(0.2, 0.5)
            r_min = float(np.random.choice(rmin_options))
            force_dist = np.random.uniform(0.0, 1.0)
            seed = np.random.randint(0, 100000) # Seed casuale per ogni processo
            tasks.append((res, vol_frac, r_min, force_dist, seed))

        res_data = []
        successful_samples = 0
        
        # Eseguiamo i task in parallelo
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Sottomettiamo i task
            futures = {executor.submit(generate_single_sample, task): task for task in tasks}
            
            # tqdm crea la barra di caricamento
            for future in tqdm(as_completed(futures), total=samples_per_res, desc=f"Res {res}"):
                result = future.result()
                
                if "error" not in result:
                    res_data.append(result)
                    successful_samples += 1
                    
                    # Salvataggio intermedio ogni 500 campioni per sicurezza
                    if successful_samples % 500 == 0:
                        np.savez_compressed(f"{save_dir}/res_{res}_checkpoint_{successful_samples}.npz", data=res_data)
                else:
                    print(f"\nErrore durante la generazione di un campione a res {res}: {result['error']}")

        # Salvataggio finale per la risoluzione
        print(f"\nSalvataggio finale per risoluzione {res} in corso...")
        np.savez_compressed(f"{save_dir}/resolution_{res}_final.npz", data=res_data)
        print(f"Risoluzione {res} completata con {successful_samples}/{samples_per_res} campioni validi.")

if __name__ == "__main__":
    # Su Windows, il multiprocessing richiede che il codice principale sia sotto l'if __name__ == "__main__"
    main()