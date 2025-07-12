#!/bin/bash
#SBATCH --job-name RedMapper
#SBATCH --output=/home/dhayaa/DECADE/RedmapperDecade/catalog/log_RedMapper
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

#conda activate /project2/chihway/dhayaa/SPT/envs/ymap/
#module load openmpi

python -u /home/dhayaa/DECADE/RedmapperDecade/catalog/Runner.py