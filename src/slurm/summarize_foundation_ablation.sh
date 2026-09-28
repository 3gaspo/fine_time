#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/foundation_ablation_schedule.sh"

experiment="${TIME_EXPERIMENT:?TIME_EXPERIMENT must name the ablation}"
case "$experiment" in
    context_size|instance_normalization) ;;
    *) echo "Unsupported foundation ablation: $experiment" >&2; exit 2 ;;
esac

TIME_WORKFLOW_NAME="${experiment}_summary"
TIME_TASK_NAME=macro_mase_timing_and_plots
TIME_STATUS_NAME=summary
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
status_root="$TIME_LOGS/$experiment/workflow_status/$experiment"
incomplete=()
for model in "${FOUNDATION_ABLATION_MODELS[@]}"; do
    if [ "$experiment" = context_size ]; then
        for context_length in $(foundation_context_lengths "$model"); do
            expected="${model}_context_${context_length}"
            state="$(sed -n 's/^state=//p' "$status_root/${TIME_LAUNCH_ID}__${expected}.status" 2>/dev/null || true)"
            exit_code="$(sed -n 's/^exit_code=//p' "$status_root/${TIME_LAUNCH_ID}__${expected}.status" 2>/dev/null || true)"
            [ "$state" = completed ] && [ "$exit_code" = 0 ] || incomplete+=("$expected")
        done
    else
        for normalization in "${FOUNDATION_NORMALIZATION_MODES[@]}"; do
            expected="${model}_normalization_${normalization}"
            state="$(sed -n 's/^state=//p' "$status_root/${TIME_LAUNCH_ID}__${expected}.status" 2>/dev/null || true)"
            exit_code="$(sed -n 's/^exit_code=//p' "$status_root/${TIME_LAUNCH_ID}__${expected}.status" 2>/dev/null || true)"
            [ "$state" = completed ] && [ "$exit_code" = 0 ] || incomplete+=("$expected")
        done
    fi
done
if [ "${#incomplete[@]}" -gt 0 ]; then
    echo "Incomplete $experiment jobs: ${incomplete[*]}" >&2
    exit 1
fi

tasks_root="$TIME_OUTPUTS/$experiment/tasks"
summary_root="$TIME_OUTPUTS/$experiment/reports"
extra_artifacts=()
config_policy=distinct

if [ "$experiment" = context_size ]; then
    time_stage_start context_horizon_plot
    context_plot="$summary_root/performance/context_horizon_mase"
    time_task_start "context_horizon_grid output=$context_plot"
    plot_command=(
        uv run --no-sync python "$PROJECT_ROOT/scripts/plot_context_size.py"
        --tasks-root "$tasks_root"
        --seasonal-root "$TIME_SEASONAL_EVALUATIONS_ROOT"
        --launch-id "$TIME_LAUNCH_ID"
        --models "${FOUNDATION_ABLATION_MODELS[@]}"
        --output "$context_plot"
    )
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 "${plot_command[@]}"
    else
        "${plot_command[@]}"
    fi
    extra_artifacts=(
        "$context_plot.csv"
        "$context_plot.png"
        "$context_plot.pdf"
    )
    time_task_complete
    time_stage_complete
fi

time_stage_start summarize
time_task_start "foundation_ablation experiment=$experiment outputs=$summary_root"
summary_command=(
    uv run --no-sync python "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
    --results-dir "$tasks_root"
    --seasonal-naive-results-dir "$TIME_SEASONAL_EVALUATIONS_ROOT"
    --models "${FOUNDATION_ABLATION_MODELS[@]}"
    --launch-id "$TIME_LAUNCH_ID"
    --config-policy "$config_policy"
    --repeat-policy latest
    --csv "$summary_root/foundation_model_summary.csv"
    --markdown "$summary_root/foundation_model_summary.md"
)
for model in "${FOUNDATION_ABLATION_MODELS[@]}"; do
    summary_command+=(--model-status "$model=completed,0")
done
for artifact in "${extra_artifacts[@]}"; do
    summary_command+=(--extra-artifact "$artifact")
done
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${summary_command[@]}"
else
    "${summary_command[@]}"
fi
if [ "$experiment" = context_size ]; then
    cat >> "$summary_root/foundation_model_summary.md" <<'EOF'

## Context size by forecast horizon

The figure places forecast horizon size on the x-axis and maximum context size
on the y-axis. Color is the geometric mean task MASE divided by the matching
Seasonal Naive MASE for each model/context/horizon-size cell.

![Context size by forecast horizon](performance/context_horizon_mase.png)
EOF
fi
time_task_complete
time_stage_complete
time_workflow_complete
