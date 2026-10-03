#!/bin/bash
#SBATCH --job-name=lora_summary
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dgx/task_finetuning_lora/slurm/%x_%j.out
#SBATCH --error=logs/dgx/task_finetuning_lora/slurm/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
export TIME_FINE_TIME_VARIANT=lora
source "$PROJECT_ROOT/src/slurm/summarize_fine_time.sh"
