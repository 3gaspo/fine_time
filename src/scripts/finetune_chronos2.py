#!/usr/bin/env python3
"""Fine-tune Chronos-2 independently on each selected TIME training split.

This is a thin project adapter around the official ``Chronos2Pipeline.fit``
implementation shipped by chronos-forecasting 2.3.1:
https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from omegaconf import DictConfig, OmegaConf

from timebench.paths import foundation_weight_path
from timebench.pipeline.runtime_resources import log_selected_device
from timebench.training import (
    allocate_training_run,
    checkpoint_artifacts,
    checkpoint_path,
    load_training_series,
    model_training_config,
    output_root,
    select_tasks,
    set_seed,
)


@hydra.main(version_base=None, config_path="../timebench/conf", config_name="fine_time")
def main(cfg: DictConfig) -> None:
    if cfg.model != "chronos2":
        raise ValueError("finetune_chronos2.py requires model=chronos2")
    values = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(values, dict)
    model_cfg = model_training_config(values, "chronos2")
    root = output_root(values.get("output_root"))
    tasks = select_tasks(
        values.get("config_path"),
        values["experiment"]["excluded_datasets"],
        values.get("dataset"),
        values.get("term"),
        int(values["experiment"]["expected_tasks"]),
    )
    base_checkpoint = foundation_weight_path(
        model_cfg["weight"], directory=True
    )
    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    for index, task in enumerate(tasks, 1):
        with allocate_training_run(root, values, "chronos2", task, base_checkpoint) as run:
            if not run.should_run:
                if run.action == "finalize":
                    run.complete()
                print(f"[{index}/{len(tasks)}] skip completed {task.key}")
                continue
            destination = checkpoint_path(run.run_dir, "chronos2")
            print(f"[{index}/{len(tasks)}] fine-tune chronos2 on {task.key}")
            set_seed(int(values["seed"]))
            series = load_training_series(task, int(values["training"]["min_context"]))
            started = time.perf_counter()
            log_selected_device(device_map, stage="finetune", model="chronos2")
            pipeline = BaseChronosPipeline.from_pretrained(
                str(base_checkpoint), device_map=device_map, local_files_only=True
            )
            if not isinstance(pipeline, Chronos2Pipeline):
                raise TypeError(f"Expected Chronos2Pipeline, got {type(pipeline).__name__}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            pipeline.fit(
                inputs=series,
                prediction_length=task.prediction_length,
                finetune_mode=str(values["training"]["mode"]),
                context_length=int(model_cfg["context_length"]),
                learning_rate=float(model_cfg["learning_rate"]),
                num_steps=int(model_cfg["steps"]),
                batch_size=int(model_cfg["batch_size"]),
                output_dir=destination.parent,
                min_past=int(values["training"]["min_context"]),
                finetuned_ckpt_name=destination.name,
                disable_data_parallel=True,
                seed=int(values["seed"]),
                data_seed=int(values["seed"]),
                max_grad_norm=float(values["training"]["gradient_clip_norm"]),
            )
            metrics_path = run.run_dir / "training_metrics.json"
            metrics_path.write_text(
                json.dumps({"training_seconds": time.perf_counter() - started}, indent=2) + "\n",
                encoding="utf-8",
            )
            run.complete(
                ["training_metrics.json", *checkpoint_artifacts(run.run_dir, destination)],
                artifact_metadata={
                    "training": {
                        "implementation": "official Chronos2Pipeline.fit",
                        "training_series": len(series),
                        "checkpoint": str(destination.relative_to(run.run_dir)),
                    }
                },
            )


if __name__ == "__main__":
    main()
