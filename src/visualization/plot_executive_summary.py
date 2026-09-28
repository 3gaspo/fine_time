"""Regenerate evaluating_tsfms executive-summary figures from selected artifacts.

Run from the project root:
    python src/visualization/plot_executive_summary.py
Use --figures-dir latex to explicitly refresh the LaTeX figure assets.
No forecasting, metric-array evaluation, or PDF compilation is performed.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timebench.visualization.performance import plot_accuracy_time, plot_feature_scatter
from plot_foundation_task_dispersion import MODELS, plot_dispersion


PROJECT = Path(__file__).resolve().parents[2]
MODES = ("multivariate", "univariate", "covariate")


def load_channel_tasks(report: Path, tasks_root: Path, mode: str):
    """Read only the completed tasks explicitly selected by a mode's report."""
    provenance = json.loads(report.read_text(encoding="utf-8"))
    rows = []
    for relative in provenance["input_manifests"]:
        manifest_text = str(relative).replace("\\", "/")
        marker = f"/outputs/channels_comparison/tasks/{mode}/"
        if marker in manifest_text:
            manifest_text = manifest_text.split(marker, 1)[1]
        manifest_path = tasks_root / manifest_text
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(manifest_path.with_name("metrics_summary.json").read_text(encoding="utf-8"))
        metric, identity = summary["metrics"]["MASE"], manifest["identity"]
        if (manifest["status"] != "completed"
                or summary["evaluation_grid"]["definition"] != "finite_ground_truth_and_seasonal_naive_mase"
                or not 0 < metric["finite_values"] <= metric["evaluation_values"]):
            raise ValueError(f"Task does not satisfy the current evaluation contract: {manifest_path}")
        rows.append({
            "mode": mode, "dataset": identity["dataset"],
            "frequency": identity["frequency"], "term": identity["term"],
            "MASE": float(metric["mean"]), "cells": metric["finite_values"],
            "grid": summary["evaluation_grid"]["source"].removeprefix("/fslarge"),
            "manifest": str(manifest_path),
        })
    frame = pd.DataFrame(rows)
    keys = ["dataset", "frequency", "term"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError(f"Empty or ambiguous channel task selection: {report}")
    return frame.set_index(keys)


def channel_ratios(frames):
    """Pair identical tasks/support across modes; never silently drop tasks."""
    native = frames["multivariate"]
    result = []
    for mode in ("univariate", "covariate"):
        frame = frames[mode]
        if set(frame.index) != set(native.index):
            raise ValueError(f"Channel task coverage differs for {mode}")
        frame = frame.reindex(native.index)
        if not frame["grid"].equals(native["grid"]) or not frame["cells"].equals(native["cells"]):
            raise ValueError(f"Channel evaluation support differs for {mode}")
        ratios = frame["MASE"] / native["MASE"]
        if not np.isfinite(ratios).all():
            raise ValueError(f"Undefined channel MASE ratios for {mode}")
        for key, ratio in ratios.items():
            result.append(dict(zip(("dataset", "frequency", "term"), key),
                               mode=mode, native_MASE=float(native.loc[key, "MASE"]),
                               alternative_MASE=float(frame.loc[key, "MASE"]), ratio=float(ratio)))
    return pd.DataFrame(result)


def plot_channel_ratios(ratios: pd.DataFrame, path: Path):
    """Box plots of task MASE/native MASE, with 5th–95th percentile whiskers."""
    figure, axis = plt.subplots(figsize=(8.5, 3.5), constrained_layout=True)
    labels = {"univariate": "Independent univariate", "covariate": "Past-target covariates"}
    modes = tuple(labels)
    values = [ratios.loc[ratios["mode"] == mode, "ratio"].to_numpy() for mode in modes]
    boxes = axis.boxplot(values, whis=(5, 95), patch_artist=True)
    for box, color in zip(boxes["boxes"], ("#669966", "#ffa02f")):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    axis.set_xticks([1, 2], [f"{labels[mode]}\n({len(value)} tasks)"
                                     for mode, value in zip(modes, values)])
    axis.axhline(1, linestyle="--", color="#555555", linewidth=1)
    axis.set_ylabel("Task MASE / native multivariate MASE")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=220)
    if Path(path).suffix.lower() == ".png":
        figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    snapshot = PROJECT / "outputs/selena"
    parser.add_argument("--foundation-report", type=Path, default=snapshot / "foundation_models/reports/foundation_model_report_manifest.json")
    parser.add_argument("--channel-summary-root", type=Path, default=snapshot / "channels_comparison/reports")
    parser.add_argument("--channel-tasks-root", type=Path, default=snapshot / "channels_comparison/tasks")
    parser.add_argument("--feature-data", type=Path, default=snapshot / "foundation_models/reports/feature_analysis/mase_vs_features_data.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/dgx/analysis/executive_summary")
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.figures_dir or args.output
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    report = json.loads(args.foundation_report.read_text(encoding="utf-8"))
    summary_path = args.foundation_report.with_name("foundation_model_summary.csv")
    plot_accuracy_time(pd.read_csv(summary_path), figures / "executive_summary_accuracy_time.png",
                       styles=MODELS, model_column="base_model")
    performance = args.foundation_report.parent / "performance"
    for suffix in (".png", ".pdf"):
        shutil.copyfile(performance / f"task_mean_std{suffix}",
                        figures / f"executive_summary_task_mean_std{suffix}")
    frames = {
        mode: load_channel_tasks(args.channel_summary_root / mode / "foundation_model_report_manifest.json",
                                 args.channel_tasks_root / mode, mode)
        for mode in MODES
    }
    ratios = channel_ratios(frames)
    ratios.to_csv(args.output / "channel_task_ratios.csv", index=False)
    plot_channel_ratios(ratios, figures / "executive_summary_channel_ratios.png")
    plot_feature_scatter(pd.read_csv(args.feature_data), figures / "executive_summary_feature_scatter.png",
                         styles=MODELS, model_column="base_model")
    plot_dispersion(PROJECT, args.foundation_report, args.output / "task_dispersion",
                    figures / "executive_summary_task_dispersion.png")
    evidence = {
        "foundation_report": str(args.foundation_report),
        "foundation_input_manifests": report["input_manifests"],
        "foundation_summary": str(summary_path),
        "channel_reports": [str(args.channel_summary_root / mode / "foundation_model_report_manifest.json") for mode in MODES],
        "channel_input_manifests": {mode: frames[mode]["manifest"].tolist() for mode in MODES},
        "feature_data": str(args.feature_data),
        "figures_dir": str(figures),
        "inference_rerun": False,
    }
    (args.output / "plot_inputs.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Four executive-summary figures saved in {figures}")


if __name__ == "__main__":
    main()
