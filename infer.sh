#!/bin/bash --login
#SBATCH -J RM # job name
#SBATCH -o o.%x.%j        # output file
#SBATCH -e e.%x.%j        # error file
#SBATCH -p gpu_v100            # partition
#SBATCH --gres=gpu:1
#SBATCH -n 8              # number of tasks (1 CPU per task by default)
#SBATCH --time=01:00:00   # time
#SBATCH --account=scw2050 # project account number

git pull origin dev-train 
module purge
module load deepspeed
module list
export PYTHONPATH="${PYTHONPATH}:/home/c.scmse/Funtuner"
exec singularity exec --nv $DEEPSPEED_IMAGE /nfshome/store03/users/c.scmse/venv/bin/python funtuner/sampling.py --model_url shahules786/Redpajama-3B-CoT --dataset Dahoas/cot_gsm8k 
exec singularity exec --nv $DEEPSPEED_IMAGE /nfshome/store03/users/c.scmse/venv/bin/python evals/sampler.py --model_url shahules786/Redpajama-3B-CoT 

