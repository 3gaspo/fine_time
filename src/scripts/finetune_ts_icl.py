#!/usr/bin/env python3
"""Fine-tune the TS-ICL forecaster on each selected TIME training split.

TS-ICL 0.2.1 exposes no public fine-tuning command. This project-specific
adapter preserves the package's input preparation, normalization, network and
checkpoint format, and optimizes its forecasting quantile head directly:
https://github.com/EDF-Lab/ts-icl/blob/main/src/tsicl/pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tsicl import TSICL
from tsicl.utils import complete_nans, make_grid

from timebench.paths import foundation_weight_path
from timebench.pipeline.runtime_resources import log_selected_device
from timebench.training import (
    allocate_training_run,
    apply_lora,
    checkpoint_artifacts,
    checkpoint_path,
    load_training_series,
    merge_lora,
    model_training_config,
    output_root,
    sample_window_batch,
    select_tasks,
    set_seed,
)
from timebench.training.ts_icl_compat import prepare_training_contexts


@hydra.main(version_base=None, config_path="../timebench/conf", config_name="fine_time")
def main(cfg: DictConfig) -> None:
    if cfg.model != "ts_icl":
        raise ValueError("finetune_ts_icl.py requires model=ts_icl")
    values = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(values, dict)
    model_cfg = model_training_config(values, "ts_icl")
    root = output_root(values.get("output_root"))
    tasks = select_tasks(
        values.get("config_path"),
        values["experiment"]["excluded_datasets"],
        values.get("dataset"),
        values.get("term"),
        int(values["experiment"]["expected_tasks"]),
    )
    base_checkpoint = foundation_weight_path(model_cfg["weight"], directory=False)

    for index, task in enumerate(tasks, 1):
        with allocate_training_run(root, values, "ts_icl", task, base_checkpoint) as run:
            if not run.should_run:
                if run.action == "finalize":
                    run.complete()
                print(f"[{index}/{len(tasks)}] skip completed {task.key}")
                continue
            destination = checkpoint_path(run.run_dir, "ts_icl")
            print(f"[{index}/{len(tasks)}] fine-tune ts_icl on {task.key}")
            generator = set_seed(int(values["seed"]))
            pipeline = TSICL(model_path=base_checkpoint, allow_auto_download=False)
            context_length = min(int(model_cfg["context_length"]), pipeline.max_context_length)
            horizon = min(task.prediction_length, pipeline.max_target_length)
            series = load_training_series(task, int(values["training"]["min_context"]))
            started = time.perf_counter()
            losses = _fit(
                pipeline, series, context_length, horizon, values, model_cfg, generator
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config": pipeline.model_config,
                    "forecaster": pipeline.forecaster.state_dict(),
                    "imputer": pipeline.imputer.state_dict(),
                },
                destination,
            )
            metrics = {
                "first_loss": losses[0],
                "final_loss": losses[-1],
                "training_seconds": time.perf_counter() - started,
            }
            (run.run_dir / "training_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            run.complete(
                ["training_metrics.json", *checkpoint_artifacts(run.run_dir, destination)],
                artifact_metadata={
                    "training": {
                        "implementation": "project adapter over TS-ICL forecasting internals",
                        "training_series": len(series),
                        "native_training_horizon": horizon,
                        "context_length": context_length,
                        "checkpoint": str(destination.relative_to(run.run_dir)),
                    }
                },
            )


def _fit(pipeline, series, context_length, horizon, values, model_cfg, generator):
    mode = str(values["training"]["mode"])
    if mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported TS-ICL fine-tuning mode: {mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_selected_device(str(device), stage="finetune", model="ts_icl")
    model = pipeline.forecaster.to(device)
    start = float(model.tf_icl.start_quantile)
    end = float(model.tf_icl.end_quantile)
    count = int(model.tf_icl.nb_quantiles)
    if mode == "lora":
        model = apply_lora(model, values, "ts_icl", _lora_target_modules(model))
        pipeline.forecaster = model
    model.train()
    pipeline._device = device
    pipeline.this_context_length = context_length
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(model_cfg["learning_rate"]),
    )
    quantiles = torch.linspace(start, end, count, device=device).view(1, 1, -1)
    losses: list[float] = []
    for _ in range(int(model_cfg["steps"])):
        context, target = sample_window_batch(
            series,
            batch_size=int(model_cfg["batch_size"]),
            context_length=context_length,
            min_context=int(values["training"]["min_context"]),
            horizon=horizon,
            generator=generator,
        )
        context = context.unsqueeze(-1).to(device)
        target = target.unsqueeze(-1).to(device)
        grid = make_grid(pipeline._grid_len_forecasting, num_samples=len(context)).to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = _run_forward(
            pipeline,
            grid,
            context,
            horizon,
        )
        normalized_target = pipeline.scaler.transform(target)
        mask = torch.isfinite(normalized_target).expand_as(predictions)
        errors = torch.nan_to_num(normalized_target, nan=0.0) - predictions
        pinball = 2 * torch.maximum(quantiles * errors, (quantiles - 1) * errors)
        loss = pinball[mask].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(values["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    if mode == "lora":
        pipeline.forecaster = merge_lora(model)
        pipeline.forecaster.eval()
    return losses


def _lora_target_modules(model) -> list[str]:
    attention = [
        name
        for name, module in model.named_modules()
        if name.startswith("tf_icl.tf_icl.")
        and isinstance(module, torch.nn.MultiheadAttention)
    ]
    quantile_outputs = [
        name
        for name, module in model.named_modules()
        if name.startswith("tf_icl.decoder.") and isinstance(module, torch.nn.Linear)
    ]
    if not attention or not quantile_outputs:
        raise ValueError("Could not resolve TS-ICL attention and quantile-output LoRA targets")
    return [*attention, quantile_outputs[-1]]


def _run_forward(pipeline, grid, context, horizon):
    """Run the differentiable TS-ICL path with singleton-safe preparation."""

    lookback_length = min(context.shape[1], pipeline.this_context_length)
    grid_threshold = pipeline.max_context_length
    coords_c = grid[:, grid_threshold - lookback_length : grid_threshold]
    coords_t = grid[:, grid_threshold : grid_threshold + horizon]
    prepared = prepare_training_contexts(
        coords_c,
        context[:, -lookback_length:],
        complete_nans,
    )
    return pipeline._predict_batch(
        series_c=prepared["series_c"],
        coords_c=prepared["coords_c"],
        coords_t=coords_t,
        setting="forecasting",
        denormalize=False,
        save_scaler=True,
    )


if __name__ == "__main__":
    main()
