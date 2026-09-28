#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/fine_time.sh dgx|selena" >&2
}

cluster="${1:-}"
case "$cluster" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$cluster" = selena ]; then
    export PROJECT_ROOT
    source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
else
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
    export TIME_STORAGE_ROOT
    source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
fi
mkdir -p "$TIME_LOGS/task_finetuning/slurm"

launch_id="${TIME_LAUNCH_ID:-${cluster}_fine_time_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
model_jobs=()
for model in chronos2 chronos_bolt ts_icl; do
    if [ "$cluster" = selena ]; then
        front="$PROJECT_ROOT/slurm/selena/fine_time/${model}_selena.slurm"
    else
        front="$PROJECT_ROOT/slurm/dgx/fine_time/${model}.slurm"
    fi
    job_id="$(sbatch --parsable --export="ALL,TIME_LAUNCH_ID=$launch_id" "$front")"
    job_id="${job_id%%;*}"
    model_jobs+=("$job_id")
    echo "Fine TIME submitted model=$model job_id=$job_id launch_id=$launch_id"
done

dependency="$(IFS=:; echo "${model_jobs[*]}")"
if [ "$cluster" = selena ]; then
    summary_front="$PROJECT_ROOT/slurm/selena/fine_time_summary_selena.slurm"
else
    summary_front="$PROJECT_ROOT/slurm/dgx/fine_time_summary.slurm"
fi
summary_job="$(
    sbatch --parsable --dependency="afterok:$dependency" \
        --export="ALL,TIME_LAUNCH_ID=$launch_id" "$summary_front"
)"
summary_job="${summary_job%%;*}"
echo "Fine TIME summary submitted job_id=$summary_job dependency=afterok:$dependency"
echo "launch_id=$launch_id"
