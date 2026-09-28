"""Summarize and plot the context-size by forecast-horizon experiment."""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compute_foundation_summary import _effective_cells, load_result_cells
from timebench.visualization.context_size import plot_context_horizon_grid


def context_horizon_rows(cells: list[dict], seasonal_cells: list[dict]) -> list[dict]:
    """Aggregate task-relative MASE for each model/context/horizon-size cell."""

    baseline = {
        (cell["dataset_id"], cell["horizon"]): cell["MASE"]
        for cell in _effective_cells(seasonal_cells)
    }
    grouped = defaultdict(list)
    for cell in cells:
        scientific = cell["scientific_config"]
        context_size = int(scientific["model_config"]["context_length"])
        horizon_size = int(scientific["pipeline_config"]["prediction_length"])
        denominator = baseline[(cell["dataset_id"], cell["horizon"])]
        grouped[(cell["base_model"], context_size, horizon_size)].append(
            float(cell["MASE"] / denominator)
        )

    rows = []
    for (model, context_size, horizon_size), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        scaled_mase = (None if not len(finite) else 0.0 if np.any(finite == 0)
                       else float(np.exp(np.nanmean(np.log(finite)))))
        rows.append(
            {
                "model": model,
                "context_size": context_size,
                "horizon_size": horizon_size,
                "scaled_MASE": scaled_mase,
                "tasks": len(values),
                "finite_tasks": len(finite),
            }
        )
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model", "context_size", "horizon_size", "scaled_MASE", "tasks", "finite_tasks"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--seasonal-root", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = load_result_cells(
        args.tasks_root,
        set(args.models),
        launch_id=args.launch_id,
        config_policy="distinct",
        repeat_policy="latest",
    )
    seasonal = load_result_cells(
        args.seasonal_root,
        {"seasonal_naive"},
        target_modes={"univariate"},
        config_policy="latest",
        repeat_policy="latest",
    )
    rows = context_horizon_rows(cells, seasonal)
    csv_path = args.output.with_suffix(".csv")
    write_rows(rows, csv_path)
    artifacts = plot_context_horizon_grid(rows, args.output)
    print(csv_path)
    for artifact in artifacts:
        print(artifact)


if __name__ == "__main__":
    main()
