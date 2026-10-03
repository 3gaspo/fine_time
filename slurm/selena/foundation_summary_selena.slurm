#!/bin/bash
#SBATCH --job-name=ssummary
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/%u/codes/fine_time/logs/foundation_models/slurm/%x_%j.out
#SBATCH --error=/scratch/users/%u/codes/fine_time/logs/foundation_models/slurm/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --exclusive
#SBATCH --wckey=P12CU:DATASCIENCE

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
export TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-selena_${SLURM_JOB_ID}}"
summary_script="${TIME_SUMMARY_SCRIPT:-summarize_foundation_models.sh}"
case "$summary_script" in
    summarize_*.sh) ;;
    *) echo "Invalid TIME_SUMMARY_SCRIPT=$summary_script" >&2; exit 2 ;;
esac
source "$PROJECT_ROOT/src/slurm/$summary_script"
