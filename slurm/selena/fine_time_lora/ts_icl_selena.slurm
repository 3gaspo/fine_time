#!/bin/bash
#SBATCH --job-name=slora_tsicl
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/%u/codes/fine_time/logs/task_finetuning_lora/slurm/%x_%j.out
#SBATCH --error=/scratch/users/%u/codes/fine_time/logs/task_finetuning_lora/slurm/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --exclusive
#SBATCH --wckey=P12CU:DATASCIENCE

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
export TIME_FINE_TIME_VARIANT=lora
export TIME_MODEL=ts_icl
source "$PROJECT_ROOT/src/slurm/run_fine_time_model.sh"
