#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --job-name=GemmProfile
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err

module load apptainer
apptainer exec --nv ../../../CuteDSL129.sif bash -c "python3 profile_script.py"