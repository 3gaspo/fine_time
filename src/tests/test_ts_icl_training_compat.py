"""Focused regression check for TS-ICL sparse-context training batches."""

import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "src/timebench/training/ts_icl_compat.py"
SPEC = importlib.util.spec_from_file_location("ts_icl_compat", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare_training_contexts = MODULE.prepare_training_contexts


def _complete(values, coords, *, is_test):
    assert is_test
    return {
        "values": torch.nan_to_num(values),
        "coords": coords,
    }


def main() -> None:
    grid = torch.arange(4, dtype=torch.float32).view(1, 4, 1).repeat(2, 1, 1)
    series = torch.tensor(
        [
            [[float("nan")], [float("nan")], [float("nan")], [1.0]],
            [[float("nan")], [float("nan")], [2.0], [3.0]],
        ]
    )

    prepared = prepare_training_contexts(grid, series, _complete)

    assert prepared["series_c"].shape == (2, 2, 1)
    assert prepared["coords_c"].shape == (2, 2, 1)
    assert prepared["series_c"][0, 0, 0] == 1.0
    assert prepared["coords_c"][0, 0, 0] == 3.0
    assert torch.isfinite(prepared["series_c"]).all()


if __name__ == "__main__":
    main()
