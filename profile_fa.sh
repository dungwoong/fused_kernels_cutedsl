#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --job-name=GemmProfile
#SBATCH --output=%j_gemm_attempt.out
#SBATCH --error=%j_gemm_attempt.err

module load apptainer
apptainer exec --nv ../rp_CuteDSL.sif bash -c "cd flashattention/attempt3_persistent && python3 profile_script.py"