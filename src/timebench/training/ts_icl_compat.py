"""TS-ICL tensor preparation used by the project-specific training adapter."""

from __future__ import annotations

from collections.abc import Callable

import torch


def prepare_training_contexts(
    grid: torch.Tensor,
    series: torch.Tensor,
    complete_nans: Callable,
) -> dict[str, torch.Tensor]:
    """Prepare a mixed-missingness batch without squeezing singleton contexts.

    TS-ICL 0.2.1 squeezes every dimension from the observed grid before
    padding. A sample with exactly one observation therefore becomes a scalar
    and cannot be concatenated with its one-dimensional padding. This is the
    same preparation with the observed grid explicitly kept one-dimensional.
    """

    if series.ndim != 3 or series.shape[-1] != 1:
        raise ValueError("TS-ICL training series must have shape (batch, time, 1)")
    if grid.ndim != 3 or grid.shape[0] != series.shape[0]:
        raise ValueError("TS-ICL training grid must have shape (batch, time, 1)")
    if grid.shape[1] != series.shape[1]:
        raise ValueError("TS-ICL training grid and series lengths must match")

    missing_per_sample = torch.isnan(series).sum(1).squeeze(-1)
    max_context_length = int(series.shape[1] - missing_per_sample.min())
    prepared_values: list[torch.Tensor] = []
    prepared_coords: list[torch.Tensor] = []

    for index in range(len(series)):
        sample = series[index].squeeze(-1)
        missing = torch.isnan(sample)
        observed_values = sample[~missing].reshape(-1)
        observed_coords = grid[index][~missing].reshape(-1)
        padding_length = max_context_length - len(observed_values)

        if padding_length:
            values = torch.cat(
                [
                    observed_values,
                    torch.full(
                        (padding_length,),
                        torch.nan,
                        device=sample.device,
                        dtype=sample.dtype,
                    ),
                ]
            ).unsqueeze(-1)
            coords = torch.cat(
                [
                    observed_coords,
                    torch.ones(
                        padding_length,
                        device=grid.device,
                        dtype=grid.dtype,
                    ),
                ]
            ).unsqueeze(-1)
            completed = complete_nans(values, coords, is_test=True)
            values, coords = completed["values"], completed["coords"]
        else:
            values = observed_values.unsqueeze(-1)
            coords = observed_coords.unsqueeze(-1)

        prepared_values.append(values.reshape(-1, 1))
        prepared_coords.append(coords.reshape(-1, 1))

    return {
        "series_c": torch.stack(prepared_values),
        "coords_c": torch.stack(prepared_coords),
    }
