#!/bin/bash
#SBATCH --job-name=gen_dataset_beams3d  # Nome aggiornato per il job
#SBATCH --time=24:00:00                 # Tempo massimo concesso (24 ore, necessario per 15.000 campioni totali)
#SBATCH --ntasks=1                      # Un singolo task
#SBATCH --cpus-per-task=8               # 8 processori per il multiprocessing
#SBATCH --mem-per-cpu=8G                # 8GB di RAM per CPU (totale 64GB)
#SBATCH --output=log_out_%j.txt         # I print e la progress bar (tqdm) finiranno qui
#SBATCH --error=log_err_%j.txt          # Gli errori finiranno qui

# 1. Carica i moduli necessari
module load stack/2024-06
module load gcc/12.2.0 python/3.11.6

# 2. Attiva il tuo ambiente virtuale
source venv/bin/activate

# 3. Lancia il generatore di dataset
# xvfb-run serve sempre perché le librerie 3D (come napari) richiedono un display virtuale
python -m engibench.problems.beams3d.generate_dataset