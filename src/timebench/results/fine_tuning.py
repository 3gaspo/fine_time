"""Paired frozen-versus-task-fine-tuned TIME result reporting."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from timebench.pipeline import manifest_reference, select_completed_runs
from timebench.results.performance import write_table


STATES = ("frozen", "task_finetuned")
COLORS = {"chronos_bolt": "#4477AA", "chronos2": "#EE7733", "ts_icl": "#228833"}


def build_fine_tuning_report(
    task_root: Path,
    destination: Path,
    models: Iterable[str],
    expected_tasks: set[str] | None = None,
) -> list[Path]:
    """Write paired results, training provenance, timings, and editable plots."""

    rows = []
    models = tuple(models)
    for state in STATES:
        selected = select_completed_runs(
            task_root / state,
            models=set(models),
            target_modes={"univariate"},
            config_policy="error",
            repeat_policy="latest",
        )
        for run_dir, manifest in selected:
            summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            mase = summary.get("metrics", {}).get("MASE", {}).get("mean")
            rows.append(
                {
                    "model": manifest["identity"]["model"],
                    "state": state,
                    "task": summary["dataset_config"],
                    "dataset": manifest["identity"]["dataset"],
                    "frequency": manifest["identity"]["frequency"],
                    "term": manifest["identity"]["term"],
                    "horizon_steps": int(manifest["pipeline_config"]["prediction_length"]),
                    "MASE": np.nan if mase is None else float(mase),
                    "inference_seconds": float(
                        manifest["artifact_metadata"]["evaluation"]["inference_seconds"]
                    ),
                    "manifest": str(run_dir / "manifest.json"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No completed Fine TIME results below {task_root}")
    duplicates = frame.duplicated(["model", "state", "task"], keep=False)
    if duplicates.any():
        raise ValueError("Multiple selected results exist for a model/state/task")
    if expected_tasks is not None:
        for model in models:
            for state in STATES:
                observed = set(
                    frame.loc[
                        (frame["model"] == model) & (frame["state"] == state), "task"
                    ]
                )
                if observed != expected_tasks:
                    missing = sorted(expected_tasks - observed)
                    extra = sorted(observed - expected_tasks)
                    raise ValueError(
                        f"Incomplete {model}/{state}: missing={missing}, extra={extra}"
                    )
    index = ["model", "task", "dataset", "frequency", "term", "horizon_steps"]
    paired = frame.pivot(index=index, columns="state", values="MASE")
    input_manifests = frame.pivot(index=index, columns="state", values="manifest").rename(
        columns={state: f"{state}_manifest" for state in STATES}
    )
    paired = paired.join(input_manifests).reset_index().rename_axis(columns=None)
    if np.isinf(paired[list(STATES)].to_numpy(dtype=float)).any():
        raise ValueError("Frozen and task-fine-tuned MASE may be finite or NaN, never infinite")
    paired["relative_change_percent"] = 100 * (
        paired["task_finetuned"] / paired["frozen"] - 1
    )
    paired["improvement_percent"] = -paired["relative_change_percent"]
    summary_rows = []
    for model, group in paired.groupby("model", sort=False):
        finite = np.isfinite(group[list(STATES)].to_numpy(dtype=float)).all(axis=1)
        matched = group[finite]
        frozen = float(np.nanmean(matched["frozen"])) if len(matched) else None
        tuned = float(np.nanmean(matched["task_finetuned"])) if len(matched) else None
        summary_rows.append({
            "model": model, "tasks": len(group), "finite_paired_tasks": len(matched),
            "frozen_mean_task_MASE": frozen, "task_finetuned_mean_task_MASE": tuned,
            "relative_change_percent": 100 * (tuned / frozen - 1) if frozen else None,
            "mean_task_MASE_improvement_percent": (
                100 * (1 - tuned / frozen) if frozen else None
            ),
            "mean_per_task_improvement_percent": (
                float(np.nanmean(matched["improvement_percent"]))
                if len(matched) else None
            ),
            "task_win_rate_percent": (100 * float(np.nanmean(
                matched["task_finetuned"].to_numpy() < matched["frozen"].to_numpy()))
                if len(matched) else None),
        })
    summary = pd.DataFrame(summary_rows)
    training = _training_runs(task_root.parent / "checkpoints", models, expected_tasks)
    optimizer_specs = _optimizer_specs(training, models)
    timing = _timing_summary(frame, training, summary, models)
    destination.mkdir(parents=True, exist_ok=True)
    paired.to_csv(destination / "task_results.csv", index=False)
    artifacts = [destination / "task_results.csv"]
    artifacts.extend(write_table(summary, destination / "summary"))
    artifacts.extend(write_table(optimizer_specs, destination / "optimizer_specs"))
    artifacts.extend(
        write_table(
            timing,
            destination / "training_and_inference_summary",
            columns=[
                "model",
                "tasks",
                "mean_per_task_improvement_percent",
                "mean_task_MASE_improvement_percent",
                "total_training_time",
                "total_inference_time",
                "total_training_plus_inference_time",
            ],
        )
    )
    finite_pairs = paired[np.isfinite(paired[list(STATES)].to_numpy(dtype=float)).all(axis=1)]
    if len(finite_pairs):
        heatmap_cells = (
            finite_pairs.groupby(
                ["model", "frequency", "horizon_steps"], as_index=False, sort=False
            )
            .agg(
                mean_improvement_percent=("improvement_percent", "mean"),
                paired_tasks=("task", "count"),
            )
        )
        artifacts.extend(
            write_table(
                heatmap_cells,
                destination / "improvement_by_frequency_horizon",
            )
        )
        for suffix in (".png", ".pdf"):
            path = destination / f"frozen_vs_finetuned_scatter{suffix}"
            _plot_task_scatter(finite_pairs, path)
            artifacts.append(path)
            path = destination / f"mean_task_mase_comparison{suffix}"
            _plot_summary(summary[summary["finite_paired_tasks"] > 0], path)
            artifacts.append(path)
        for model in models:
            cells = heatmap_cells[heatmap_cells["model"] == model]
            for suffix in (".png", ".pdf"):
                path = destination / f"improvement_heatmap_{model}{suffix}"
                _plot_improvement_heatmap(cells, model, path)
                artifacts.append(path)
    manifest = destination / "report_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric": "MASE; lower is better",
                "aggregation": "nanmean of finite paired task-level MASE with equal task weights",
                "comparison": "same model and same TIME task; frozen versus independently task-fine-tuned",
                "improvement_summaries": {
                    "mean_per_task_improvement_percent": "mean of 100 * (1 - task_finetuned_MASE / frozen_MASE)",
                    "mean_task_MASE_improvement_percent": "100 * (1 - mean_task_finetuned_MASE / mean_frozen_MASE)",
                },
                "selection": {
                    "target_mode": "univariate",
                    "config_policy": "error",
                    "repeat_policy": "latest",
                },
                "heatmap": {
                    "value": "100 * (1 - task_finetuned_MASE / frozen_MASE)",
                    "aggregation": "arithmetic mean across finite paired tasks in each cell",
                    "rows": "literal task sampling frequency",
                    "columns": "forecast horizon H in observations",
                },
                "timing": {
                    "training": "sum of completed_at - started_at for selected completed checkpoint manifests",
                    "inference": "sum of recorded inference_seconds for frozen and task_finetuned evaluations",
                    "interruptions": "failed or incomplete attempts are not selected or counted",
                },
                "inputs": {
                    "evaluation_dependencies": [
                        manifest_reference(path)
                        for path in sorted(frame["manifest"].tolist())
                    ],
                    "training_dependencies": [
                        manifest_reference(path)
                        for path in sorted(training["manifest"].tolist())
                    ],
                },
                "artifacts": [path.name for path in artifacts],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [*artifacts, manifest]


def _training_runs(
    checkpoint_root: Path,
    models: tuple[str, ...],
    expected_tasks: set[str] | None,
) -> pd.DataFrame:
    rows = []
    selected = select_completed_runs(
        checkpoint_root,
        models=set(models),
        config_policy="error",
        repeat_policy="latest",
    )
    for run_dir, manifest in selected:
        identity = manifest["identity"]
        task = f"{identity['dataset']}/{identity['term']}"
        config = manifest["model_config"]["training"]
        started = datetime.fromisoformat(manifest["started_at"])
        completed = datetime.fromisoformat(manifest["completed_at"])
        rows.append(
            {
                "model": identity["model"],
                "task": task,
                "training_seconds": (completed - started).total_seconds(),
                "learning_rate": float(config["learning_rate"]),
                "steps": int(config["steps"]),
                "batch_size": int(config["batch_size"]),
                "context_length": int(config["context_length"]),
                "gradient_clip_norm": float(config["gradient_clip_norm"]),
                "fine_tune_mode": str(config["mode"]),
                "manifest": str(run_dir / "manifest.json"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No completed Fine TIME checkpoints below {checkpoint_root}")
    if frame.duplicated(["model", "task"], keep=False).any():
        raise ValueError("Multiple selected checkpoints exist for a model/task")
    if expected_tasks is not None:
        for model in models:
            observed = set(frame.loc[frame["model"] == model, "task"])
            if observed != expected_tasks:
                missing = sorted(expected_tasks - observed)
                extra = sorted(observed - expected_tasks)
                raise ValueError(
                    f"Incomplete {model} training: missing={missing}, extra={extra}"
                )
    return frame


def _optimizer_specs(training: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    provenance = {
        "chronos2": {
            "trainer_provenance": "official Chronos2Pipeline.fit (chronos-forecasting 2.3.1)",
            "optimizer": "AdamW (torch fused)",
            "optimizer_hyperparameters": "betas=(0.9, 0.999); eps=1e-8; weight_decay=0.0",
            "lr_scheduler": "linear; 0 warmup steps",
            "loss": "official Chronos-2 training objective",
            "adapted_parameters": "full model",
            "gradient_accumulation_steps": 1,
        },
        "chronos_bolt": {
            "trainer_provenance": "source-adapted Fine TIME loop over official Chronos-Bolt forward loss",
            "optimizer": "AdamW (PyTorch defaults except learning rate)",
            "optimizer_hyperparameters": "betas=(0.9, 0.999); eps=1e-8; weight_decay=0.01",
            "lr_scheduler": "none",
            "loss": "native Chronos-Bolt multi-quantile loss",
            "adapted_parameters": "full model",
            "gradient_accumulation_steps": 1,
        },
        "ts_icl": {
            "trainer_provenance": "project-specific Fine TIME loop over TS-ICL 0.2.1 forecaster",
            "optimizer": "AdamW (PyTorch defaults except learning rate)",
            "optimizer_hyperparameters": "betas=(0.9, 0.999); eps=1e-8; weight_decay=0.01",
            "lr_scheduler": "none",
            "loss": "TS-ICL quantile pinball loss",
            "adapted_parameters": "forecaster only",
            "gradient_accumulation_steps": 1,
        },
    }
    rows = []
    for model in models:
        fields = [
            "learning_rate",
            "steps",
            "batch_size",
            "context_length",
            "gradient_clip_norm",
            "fine_tune_mode",
        ]
        configured = training.loc[training["model"] == model, fields].drop_duplicates()
        if len(configured) != 1:
            raise ValueError(f"Selected {model} checkpoints do not share one training config")
        values = configured.iloc[0]
        rows.append(
            {
                "model": model,
                **provenance[model],
                "fine_tune_mode": values["fine_tune_mode"],
                "learning_rate": values["learning_rate"],
                "steps": values["steps"],
                "epochs": "not defined; training is step-limited",
                "batch_size": values["batch_size"],
                "nominal_window_draws_per_task": int(values["steps"])
                * int(values["batch_size"]),
                "context_length": values["context_length"],
                "gradient_clip_norm": values["gradient_clip_norm"],
            }
        )
    return pd.DataFrame(rows)


def _timing_summary(
    evaluations: pd.DataFrame,
    training: pd.DataFrame,
    summary: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for model in models:
        train_seconds = float(
            training.loc[training["model"] == model, "training_seconds"].sum()
        )
        inference_seconds = float(
            evaluations.loc[evaluations["model"] == model, "inference_seconds"].sum()
        )
        result = summary.loc[summary["model"] == model].iloc[0]
        total = train_seconds + inference_seconds
        rows.append(
            {
                "model": model,
                "tasks": int(result["tasks"]),
                "mean_per_task_improvement_percent": result[
                    "mean_per_task_improvement_percent"
                ],
                "mean_task_MASE_improvement_percent": result[
                    "mean_task_MASE_improvement_percent"
                ],
                "total_training_seconds": train_seconds,
                "total_training_time": _format_duration(train_seconds),
                "total_inference_seconds": inference_seconds,
                "total_inference_time": _format_duration(inference_seconds),
                "total_training_plus_inference_seconds": total,
                "total_training_plus_inference_time": _format_duration(total),
            }
        )
    return pd.DataFrame(rows)


def _format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _plot_task_scatter(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    lower = float(frame[["frozen", "task_finetuned"]].min().min())
    upper = float(frame[["frozen", "task_finetuned"]].max().max())
    for model, group in frame.groupby("model", sort=False):
        color = COLORS.get(model)
        ax.scatter(
            group["frozen"], group["task_finetuned"], s=22, alpha=0.45,
            color=color, label=model.replace("_", " "),
        )
        ax.scatter(
            [group["frozen"].mean()], [group["task_finetuned"].mean()],
            s=180, marker="*", color=color, edgecolor="black", linewidth=0.7,
        )
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="0.35", linewidth=1)
    if lower > 0:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Frozen task MASE")
    ax.set_ylabel("Task-fine-tuned task MASE")
    ax.set_title("Per-task fine-tuning on TIME")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(summary))
    for index, row in summary.reset_index(drop=True).iterrows():
        color = COLORS.get(row["model"])
        ax.plot(
            [positions[index] - 0.12, positions[index] + 0.12],
            [row["frozen_mean_task_MASE"], row["task_finetuned_mean_task_MASE"]],
            color=color, linewidth=1.5,
        )
        ax.scatter(
            [positions[index] - 0.12, positions[index] + 0.12],
            [row["frozen_mean_task_MASE"], row["task_finetuned_mean_task_MASE"]],
            color=color, s=70, marker="o",
        )
    ax.set_xticks(positions, [value.replace("_", " ") for value in summary["model"]])
    ax.set_ylabel("Mean task MASE (lower is better)")
    ax.set_title("Frozen (left) and per-task fine-tuned (right)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_improvement_heatmap(frame: pd.DataFrame, model: str, path: Path) -> None:
    frequencies = sorted(frame["frequency"].unique(), key=_frequency_order)
    horizons = sorted(int(value) for value in frame["horizon_steps"].unique())
    matrix = (
        frame.pivot(
            index="frequency",
            columns="horizon_steps",
            values="mean_improvement_percent",
        )
        .reindex(index=frequencies, columns=horizons)
    )
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    span = max(float(np.max(np.abs(finite))), 1e-3)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E0E0E0")
    width = max(7.0, 1.1 * len(horizons) + 2.8)
    height = max(4.8, 0.48 * len(frequencies) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=-span, vmax=span),
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    ax.set_xticks(np.arange(len(horizons)), horizons)
    ax.set_yticks(np.arange(len(frequencies)), frequencies)
    ax.set_xlabel("Forecast horizon H (observations)")
    ax.set_ylabel("Task sampling frequency")
    ax.set_title(f"{model.replace('_', ' ')}: mean MASE improvement vs frozen")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Improvement (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _frequency_order(value: str) -> tuple[float, str]:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*([A-Za-z]+)\s*", str(value))
    if match is None:
        return (float("inf"), str(value))
    multiplier = float(match.group(1) or 1.0)
    unit = match.group(2).lower()
    seconds = {
        "s": 1,
        "sec": 1,
        "t": 60,
        "min": 60,
        "h": 3600,
        "d": 86400,
        "b": 86400,
        "w": 7 * 86400,
        "m": 30.4375 * 86400,
        "ms": 30.4375 * 86400,
        "me": 30.4375 * 86400,
        "q": 91.3125 * 86400,
        "qs": 91.3125 * 86400,
        "y": 365.25 * 86400,
        "a": 365.25 * 86400,
        "ys": 365.25 * 86400,
    }.get(unit)
    return (multiplier * seconds if seconds is not None else float("inf"), str(value))
