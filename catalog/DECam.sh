#!/bin/bash
#SBATCH --job-name RedMapperDECam
#SBATCH --output=/home/dhayaa/DECADE/RedmapperDecade/catalog/log_DECam
#SBATCH --partition=kicp
#SBATCH --account=kicp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --exclude=midway3-0222
python -u /home/dhayaa/DECADE/RedmapperDecade/catalog/Runner.py --Combined --outdir /project/kadrlica/dhayaa/Redmapper/DECam_20250828/Files/
