#!/bin/bash

set -euo pipefail

usage() { echo "usage: bash scripts/context_size.sh dgx|selena" >&2; }
cluster="${1:-}"
case "$cluster" in dgx|selena) ;; *) usage; exit 2 ;; esac

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
mkdir -p "$TIME_LOGS/context_size/slurm"
source "$PROJECT_ROOT/src/slurm/foundation_ablation_schedule.sh"

launch_id="${TIME_LAUNCH_ID:-${cluster}_context_size_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
jobs=()
for model in "${FOUNDATION_ABLATION_MODELS[@]}"; do
    for context_length in $(foundation_context_lengths "$model"); do
        if [ "$cluster" = selena ]; then
            front="$PROJECT_ROOT/slurm/selena/foundation_models/${model}_selena.slurm"
        else
            front="$PROJECT_ROOT/slurm/dgx/foundation_models/${model}.slurm"
        fi
        status_name="${model}_context_${context_length}"
        job_id="$(sbatch --parsable \
            --output="$TIME_LOGS/context_size/slurm/%x_%j.out" \
            --error="$TIME_LOGS/context_size/slurm/%x_%j.err" \
            --export="ALL,TIME_EXPERIMENT=context_size,TIME_LAUNCH_ID=$launch_id,TIME_CONTEXT_LENGTH=$context_length,TIME_INSTANCE_NORMALIZATION=none,TIME_STATUS_NAME=$status_name" \
            "$front")"
        job_id="${job_id%%;*}"
        jobs+=("$job_id")
        echo "context-size submitted model=$model context=$context_length job_id=$job_id launch_id=$launch_id"
    done
done

dependency="$(IFS=:; echo "${jobs[*]}")"
if [ "$cluster" = selena ]; then
    summary_front="$PROJECT_ROOT/slurm/selena/foundation_summary_selena.slurm"
else
    summary_front="$PROJECT_ROOT/slurm/dgx/foundation_summary.slurm"
fi
summary_job="$(sbatch --parsable --dependency="afterany:$dependency" \
    --output="$TIME_LOGS/context_size/slurm/%x_%j.out" \
    --error="$TIME_LOGS/context_size/slurm/%x_%j.err" \
    --export="ALL,TIME_EXPERIMENT=context_size,TIME_LAUNCH_ID=$launch_id,TIME_SUMMARY_SCRIPT=summarize_foundation_ablation.sh" \
    "$summary_front")"
summary_job="${summary_job%%;*}"
echo "context-size summary submitted job_id=$summary_job dependency=afterany:$dependency"
echo "launch_id=$launch_id"
