"""Plot paired model/Seasonal task MASE mean and population-variance ratios."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "chronos2": ("Chronos-2", "#005bbb", "o"),
    "ts_icl": ("TS-ICL", "#ffa02f", "s"),
    "chronos_bolt": ("Chronos-Bolt", "#669966", "^"),
}


def plot_dispersion(project: Path, report_path: Path, output: Path, figure_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tasks_root = project / "outputs/selena/foundation_models/tasks"
    seasonal_path = project / "outputs/selena/analysis/seasonal_dispersion/task_summary.csv"
    with seasonal_path.open(encoding="utf-8", newline="") as stream:
        seasonal_rows = list(csv.DictReader(stream))
    seasonal = {(r["dataset"], r["frequency"], r["term"]): r for r in seasonal_rows}
    assert len(seasonal) == len(seasonal_rows) == 98
    # /fslarge is the filesystem alias for the same selected /scratch paths.
    canonical = lambda path: path.removeprefix("/fslarge")
    assert {canonical(r["manifest"]) for r in seasonal_rows} == {
        canonical(path) for path in report["seasonal_naive_input_manifests"]}
    rows = []
    for relative in report["input_manifests"]:
        manifest_text = str(relative).replace("\\", "/")
        marker = "/outputs/foundation_models/tasks/"
        if marker in manifest_text:
            manifest_text = manifest_text.split(marker, 1)[1]
        path = tasks_root / manifest_text
        manifest = json.loads(path.read_text(encoding="utf-8"))
        summary = json.loads(path.with_name("metrics_summary.json").read_text(encoding="utf-8"))
        metric = summary["metrics"]["MASE"]
        assert manifest["status"] == "completed"
        assert summary["evaluation_grid"]["definition"] == "finite_ground_truth_and_seasonal_naive_mase"
        assert metric["dispersion_ddof"] == 0
        assert 0 < metric["finite_values"] <= metric["evaluation_values"]
        assert np.isclose(metric["std"] ** 2, metric["variance"], rtol=1e-12, atol=1e-12)
        identity = manifest["identity"]
        baseline = seasonal[(identity["dataset"], identity["frequency"], identity["term"])]
        seasonal_mean, seasonal_variance = float(baseline["mean"]), float(baseline["variance"])
        assert int(baseline["dispersion_ddof"]) == 0
        assert 0 < int(baseline["finite_values"]) <= int(baseline["evaluation_values"])
        assert canonical(baseline["evaluation_grid"]) == canonical(summary["evaluation_grid"]["source"])
        assert np.isfinite(seasonal_mean) and seasonal_mean > 0
        assert np.isfinite(seasonal_variance) and seasonal_variance > 0
        assert np.isclose(float(baseline["std"]) ** 2, seasonal_variance, rtol=1e-12, atol=1e-12)
        rows.append({
            "model": identity["model"], "dataset": identity["dataset"],
            "frequency": identity["frequency"], "term": identity["term"],
            "mean_MASE": metric["mean"], "variance_MASE": metric["variance"],
            "std_MASE": metric["std"], "finite_cells": metric["finite_values"],
            "seasonal_mean_MASE": seasonal_mean, "seasonal_variance_MASE": seasonal_variance,
            "scaled_mean_MASE": metric["mean"] / seasonal_mean,
            "relative_variance_MASE": metric["variance"] / seasonal_variance,
            "seasonal_manifest": baseline["manifest"],
            "manifest": path.relative_to(project).as_posix(),
        })
    assert len(rows) == 294
    assert len({(r["model"], r["dataset"], r["frequency"], r["term"]) for r in rows}) == 294
    output.mkdir(parents=True, exist_ok=True)
    with (output / "foundation_task_dispersion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(9.4, 3.9), constrained_layout=True)
    stats = {}
    for model, (label, color, marker) in MODELS.items():
        values = [row for row in rows if row["model"] == model]
        assert len(values) == 98
        means = np.array([row["scaled_mean_MASE"] for row in values])
        variances = np.array([row["relative_variance_MASE"] for row in values])
        finite = np.isfinite(means) & (means > 0) & np.isfinite(variances) & (variances > 0)
        plotted_means, plotted_variances = means[finite], variances[finite]
        axis.scatter(plotted_means, plotted_variances, color=color, marker=marker, s=34, alpha=0.72,
                     linewidths=0.4, edgecolors="white",
                     label=f"{label} ({len(plotted_means)}/98 finite tasks)")
        finite_means = means[np.isfinite(means) & (means > 0)]
        stats[model] = {
            "tasks": len(values), "finite_dispersion_tasks": int(finite.sum()),
            "median_relative_variance_MASE": (float(np.nanmedian(plotted_variances))
                                                if len(plotted_variances) else None),
            "variance_range": ([float(np.nanmin(plotted_variances)), float(np.nanmax(plotted_variances))]
                               if len(plotted_variances) else None),
            "lower_variance_than_seasonal": int(np.count_nonzero(plotted_variances < 1)),
            "better_mean_and_lower_variance": int(np.count_nonzero(
                (plotted_means < 1) & (plotted_variances < 1))),
            "geometric_mean_scaled_MASE": (float(np.exp(np.nanmean(np.log(finite_means))))
                                             if len(finite_means) else None),
        }
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([0.2, 0.4, 0.6, 1, 2, 4], ["0.2", "0.4", "0.6", "1", "2", "4"])
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_xlabel("Mean MASE / Seasonal mean MASE")
    axis.set_ylabel("Variance / Seasonal variance")
    axis.axvline(1, color="#555555", linestyle="--", linewidth=1)
    axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
    axis.grid(True, which="major", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True, fontsize=14)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220)
    if Path(figure_path).suffix.lower() == ".png":
        figure.savefig(Path(figure_path).with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output / "foundation_task_dispersion.png", dpi=220)
    if Path(output / "foundation_task_dispersion.png").suffix.lower() == ".png":
        figure.savefig(Path(output / "foundation_task_dispersion.png").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    evidence = {
        "report": report_path.relative_to(project).as_posix(),
        "selection": report["selection"], "models": stats,
        "points": len(rows), "population_ddof": 0,
        "seasonal_source": seasonal_path.relative_to(project).as_posix(),
        "seasonal_tasks": len(seasonal),
        "x": "model arithmetic task MASE mean divided by matched Seasonal task MASE mean",
        "y": "model population task MASE variance divided by matched Seasonal population task MASE variance",
        "parity": "1 on each axis; y is a variance ratio, not variance of scaled MASE",
        "axes": "logarithmic; all points positive, no exclusions or pseudocounts",
        "limitations": "compact summaries checked; raw metric arrays not independently inspected locally; not repeated-run uncertainty",
        "selected_manifests": [row["manifest"] for row in rows],
    }
    (output / "foundation_task_dispersion_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[2]
    parser.add_argument("--report", type=Path, default=project / "outputs/selena/foundation_models/reports/foundation_model_report_manifest.json")
    parser.add_argument("--output", type=Path, default=project / "outputs/dgx/analysis/task_dispersion")
    parser.add_argument("--figure", type=Path, default=project / "latex/executive_summary_task_dispersion.png")
    args = parser.parse_args()
    evidence = plot_dispersion(project, args.report, args.output, args.figure)
    print(json.dumps({"points": evidence["points"], "models": evidence["models"]}, indent=2))


if __name__ == "__main__":
    main()
