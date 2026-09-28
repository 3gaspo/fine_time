#!/bin/bash
#SBATCH --job-name=time_summary
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dgx/foundation_models/slurm/%x_%j.out
#SBATCH --error=logs/dgx/foundation_models/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
summary_script="${TIME_SUMMARY_SCRIPT:-summarize_foundation_models.sh}"
case "$summary_script" in
    summarize_*.sh) ;;
    *) echo "Invalid TIME_SUMMARY_SCRIPT=$summary_script" >&2; exit 2 ;;
esac
source "$PROJECT_ROOT/src/slurm/$summary_script"
