#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

model="${TIME_MODEL:?TIME_MODEL must be chronos2, chronos_bolt, or ts_icl}"
case "$model" in
    chronos2) training_script=finetune_chronos2.py ;;
    chronos_bolt) training_script=finetune_chronos_bolt.py ;;
    ts_icl) training_script=finetune_ts_icl.py ;;
    *) echo "unsupported Fine TIME model: $model" >&2; exit 2 ;;
esac

variant="${TIME_FINE_TIME_VARIANT:-full}"
case "$variant" in
    full)
        config_name=fine_time
        TIME_WORKFLOW_NAME=fine_time
        TIME_EXPERIMENT=task_finetuning
        ;;
    lora)
        config_name=fine_time_lora
        TIME_WORKFLOW_NAME=fine_time_lora
        TIME_EXPERIMENT=task_finetuning_lora
        ;;
    *) echo "unsupported Fine TIME variant: $variant" >&2; exit 2 ;;
esac
TIME_TASK_NAME="$model"
TIME_STATUS_NAME="$model"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
TIME_RESULT_SCOPE="$TIME_OUTPUTS/$TIME_EXPERIMENT"
export TIME_WORKFLOW_NAME TIME_EXPERIMENT TIME_TASK_NAME TIME_STATUS_NAME
export TIME_LAUNCH_ID TIME_RESULT_SCOPE PYTHONPATH
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

run_python() {
    local script="$1"
    shift
    local command=(
        uv run --no-sync python "$PROJECT_ROOT/src/scripts/$script"
        --config-name "$config_name" "model=$model" "$@"
    )
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 "${command[@]}"
    else
        "${command[@]}"
    fi
}

time_workflow_init
time_stage_start train
time_task_start "model=$model adaptation=$variant independent_per_task_fine_tuning=true profile=small"
run_python "$training_script"
time_task_complete
time_stage_complete

time_stage_start evaluate
time_task_start "model=$model states=frozen,task_finetuned target_mode=univariate"
run_python evaluate_fine_time.py
time_task_complete
time_stage_complete
time_workflow_complete
