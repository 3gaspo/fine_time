"""Paired frozen-versus-task-fine-tuned TIME result reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timebench.pipeline import select_completed_runs
from timebench.results.performance import write_table


STATES = ("frozen", "task_finetuned")
COLORS = {"chronos_bolt": "#4477AA", "chronos2": "#EE7733", "ts_icl": "#228833"}


def build_fine_tuning_report(
    task_root: Path,
    destination: Path,
    models: Iterable[str],
    expected_tasks: set[str] | None = None,
) -> list[Path]:
    """Write paired task values, equal-task summaries, and two editable plots."""

    rows = []
    models = tuple(models)
    for state in STATES:
        selected = select_completed_runs(
            task_root / state,
            models=set(models),
            target_modes={"univariate"},
            config_policy="latest",
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
                    "MASE": np.nan if mase is None else float(mase),
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
    paired = (
        frame.pivot(index=["model", "task"], columns="state", values="MASE")
        .reset_index()
        .rename_axis(columns=None)
    )
    if np.isinf(paired[list(STATES)].to_numpy(dtype=float)).any():
        raise ValueError("Frozen and task-fine-tuned MASE may be finite or NaN, never infinite")
    paired["relative_change_percent"] = 100 * (
        paired["task_finetuned"] / paired["frozen"] - 1
    )
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
            "task_win_rate_percent": (100 * float(np.nanmean(
                matched["task_finetuned"].to_numpy() < matched["frozen"].to_numpy()))
                if len(matched) else None),
        })
    summary = pd.DataFrame(summary_rows)
    destination.mkdir(parents=True, exist_ok=True)
    paired.to_csv(destination / "task_results.csv", index=False)
    artifacts = [destination / "task_results.csv"]
    artifacts.extend(write_table(summary, destination / "summary"))
    finite_pairs = paired[np.isfinite(paired[list(STATES)].to_numpy(dtype=float)).all(axis=1)]
    if len(finite_pairs):
        for suffix in (".png", ".pdf"):
            path = destination / f"frozen_vs_finetuned_scatter{suffix}"
            _plot_task_scatter(finite_pairs, path)
            artifacts.append(path)
            path = destination / f"mean_task_mase_comparison{suffix}"
            _plot_summary(summary[summary["finite_paired_tasks"] > 0], path)
            artifacts.append(path)
    manifest = destination / "report_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric": "MASE; lower is better",
                "aggregation": "nanmean of finite paired task-level MASE with equal task weights",
                "comparison": "same model and same TIME task; frozen versus independently task-fine-tuned",
                "artifacts": [path.name for path in artifacts],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [*artifacts, manifest]


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
