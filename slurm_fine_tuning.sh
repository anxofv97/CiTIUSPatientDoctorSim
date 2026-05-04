#!/bin/bash
#SBATCH --job-name=sim-llm           # Job name
#SBATCH --partition=LocalQ
#SBATCH --nodelist=localhost
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=4            # -c Run 1 processor per task       
#SBATCH --gres=gpu:gtx5090:1
#SBATCH --mem=50G                    # Job memory request
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --qos=regular                 # Cola
#SBATCH --output=log_%x_%j.log       # Standard output and error log

source ~/.bashrc
source eeyore_env/bin/activate
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
python fine_tuning.py --mode train --dataset annomi --num-train-epochs 1 --windows 1 5 \
 --train-frac 0.90 --no-early-stopping --require-full-window --output-base models_1e/qwen-annomi-therapist

python fine_tuning.py --mode train --dataset annomi --num-train-epochs 3 --windows 1 5 \
 --train-frac 0.90 --no-early-stopping --require-full-window --output-base models_3e/qwen-annomi-therapist
# Queue job: sbatch slurm.sh
# Check job status: squeue
# Cancel job: scancel <job_id>