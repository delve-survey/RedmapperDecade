#!/bin/bash
#SBATCH --job-name RedMapperDECADE
#SBATCH --output=/home/dhayaa/DECADE/RedmapperDecade/catalog/log_DECADE
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END

python -u /home/dhayaa/DECADE/RedmapperDecade/catalog/Runner.py --DECADE --outdir /project/kadrlica/dhayaa/Redmapper/DECADE_20250820/Files/