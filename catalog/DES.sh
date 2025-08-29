#!/bin/bash
#SBATCH --job-name RedMapperDES
#SBATCH --output=/home/dhayaa/DECADE/RedmapperDecade/catalog/log_DES
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway3-0222
python -u /home/dhayaa/DECADE/RedmapperDecade/catalog/Runner.py --DES --outdir /project/kadrlica/dhayaa/Redmapper/DES_20250828/Files/
