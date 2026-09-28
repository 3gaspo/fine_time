"""Context-size experiment visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_context_horizon_grid(rows: list[dict], output: str | Path) -> list[Path]:
    """Plot horizon on x, context size on y, and scaled MASE as color."""

    rows = [row for row in rows if row["scaled_MASE"] is not None
            and np.isfinite(row["scaled_MASE"])]
    models = sorted({row["model"] for row in rows})
    if not models:
        return []
    values = np.asarray([row["scaled_MASE"] for row in rows], dtype=float)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1e-12

    figure, axes = plt.subplots(
        len(models), 1, figsize=(10, 3.2 * len(models)), squeeze=False,
        constrained_layout=True,
    )
    marker = None
    for axis, model in zip(axes[:, 0], models):
        selected = [row for row in rows if row["model"] == model]
        marker = axis.scatter(
            [row["horizon_size"] for row in selected],
            [row["context_size"] for row in selected],
            c=[row["scaled_MASE"] for row in selected],
            s=85,
            marker="s",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_title(model)
        axis.set_xlabel("Forecast horizon size")
        axis.set_ylabel("Maximum context size")
        axis.grid(alpha=0.2)
    figure.colorbar(marker, ax=axes[:, 0].tolist(), label="Scaled MASE")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]
