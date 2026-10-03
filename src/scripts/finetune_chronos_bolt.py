#!/usr/bin/env python3
"""Fine-tune Chronos-Bolt independently on each selected TIME training split.

Chronos-Bolt has no ``Chronos2Pipeline.fit`` equivalent in
chronos-forecasting 2.3.1. This source-adapted path calls the official model's
native multi-quantile loss directly:
https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from chronos import BaseChronosPipeline, ChronosBoltPipeline
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(version_base=None, config_path="../timebench/conf", config_name="fine_time")
def main(cfg: DictConfig) -> None:
    if cfg.model != "chronos_bolt":
        raise ValueError("finetune_chronos_bolt.py requires model=chronos_bolt")
    values = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(values, dict)
    model_cfg = model_training_config(values, "chronos_bolt")
    root = output_root(values.get("output_root"))
    tasks = select_tasks(
        values.get("config_path"),
        values["experiment"]["excluded_datasets"],
        values.get("dataset"),
        values.get("term"),
        int(values["experiment"]["expected_tasks"]),
    )
    base_checkpoint = foundation_weight_path(model_cfg["weight"], directory=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    for index, task in enumerate(tasks, 1):
        with allocate_training_run(root, values, "chronos_bolt", task, base_checkpoint) as run:
            if not run.should_run:
                if run.action == "finalize":
                    run.complete()
                print(f"[{index}/{len(tasks)}] skip completed {task.key}")
                continue
            destination = checkpoint_path(run.run_dir, "chronos_bolt")
            print(f"[{index}/{len(tasks)}] fine-tune chronos_bolt on {task.key}")
            generator = set_seed(int(values["seed"]))
            log_selected_device(device_map, stage="finetune", model="chronos_bolt")
            pipeline = BaseChronosPipeline.from_pretrained(
                str(base_checkpoint), device_map=device_map, local_files_only=True
            )
            if not isinstance(pipeline, ChronosBoltPipeline):
                raise TypeError(f"Expected ChronosBoltPipeline, got {type(pipeline).__name__}")
            horizon = min(task.prediction_length, pipeline.model_prediction_length)
            series = load_training_series(task, int(values["training"]["min_context"]))
            started = time.perf_counter()
            losses = _fit(pipeline, series, horizon, values, model_cfg, generator)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pipeline.model.save_pretrained(destination)
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
                        "implementation": "source-adapted native Chronos-Bolt quantile loss",
                        "training_series": len(series),
                        "native_training_horizon": horizon,
                        "checkpoint": str(destination.relative_to(run.run_dir)),
                    }
                },
            )


def _fit(pipeline, series, horizon, values, model_cfg, generator) -> list[float]:
    mode = str(values["training"]["mode"])
    if mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported Chronos-Bolt fine-tuning mode: {mode}")
    model = pipeline.model
    if mode == "lora":
        model = apply_lora(model, values, "chronos_bolt")
        pipeline.model = model
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(model_cfg["learning_rate"]),
    )
    device = next(model.parameters()).device
    losses: list[float] = []
    for _ in range(int(model_cfg["steps"])):
        context, target = sample_window_batch(
            series,
            batch_size=int(model_cfg["batch_size"]),
            context_length=int(model_cfg["context_length"]),
            min_context=int(values["training"]["min_context"]),
            horizon=horizon,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            context=context.to(device),
            target=target.to(device),
        )
        if output.loss is None:
            raise RuntimeError("Chronos-Bolt did not return its native training loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(values["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(output.loss.detach().cpu()))
    model.eval()
    if mode == "lora":
        pipeline.model = merge_lora(model)
        pipeline.model.eval()
    return losses


if __name__ == "__main__":
    main()
