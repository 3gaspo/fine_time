#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

TIME_WORKFLOW_NAME=fine_time_summary
TIME_EXPERIMENT=task_finetuning
TIME_TASK_NAME=frozen_vs_task_finetuned
TIME_STATUS_NAME=summary
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
TIME_RESULT_SCOPE="$TIME_OUTPUTS/task_finetuning"
export TIME_WORKFLOW_NAME TIME_EXPERIMENT TIME_TASK_NAME TIME_STATUS_NAME
export TIME_LAUNCH_ID TIME_RESULT_SCOPE PYTHONPATH
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

command=(uv run --no-sync python "$PROJECT_ROOT/src/scripts/report_fine_time.py")
time_workflow_init
time_stage_start summarize
time_task_start "equal_task_weighted_MASE states=frozen,task_finetuned"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${command[@]}"
else
    "${command[@]}"
fi
time_task_complete
time_stage_complete
time_workflow_complete
