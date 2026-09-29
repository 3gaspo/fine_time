#!/usr/bin/env python3
"""Evaluate frozen and independently task-fine-tuned models on one TIME grid."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chronos2 import run_chronos2_experiment
from experiments.chronos_bolt import run_chronos_bolt_experiment
from experiments.ts_icl import run_tsicl_experiment
from timebench.paths import foundation_weight_path
from timebench.training import (
    FINE_TIME_MODELS,
    checkpoint_path,
    model_training_config,
    output_root,
    select_training_run,
    select_tasks,
)


@hydra.main(version_base=None, config_path="../timebench/conf", config_name="fine_time")
def main(cfg: DictConfig) -> None:
    values = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(values, dict)
    model = str(values["model"])
    if model not in FINE_TIME_MODELS:
        raise ValueError(f"model must be one of {FINE_TIME_MODELS}")
    model_cfg = model_training_config(values, model)
    tasks = select_tasks(
        values.get("config_path"),
        values["experiment"]["excluded_datasets"],
        values.get("dataset"),
        values.get("term"),
        int(values["experiment"]["expected_tasks"]),
    )
    root = output_root(values.get("output_root"))
    base_checkpoint = foundation_weight_path(
        model_cfg["weight"], directory=model != "ts_icl"
    )
    config_path = Path(values["config_path"]) if values.get("config_path") else None
    previous_experiment = os.environ.get("TIME_EXPERIMENT")
    os.environ["TIME_EXPERIMENT"] = "task_finetuning"
    try:
        for index, task in enumerate(tasks, 1):
            training_run, training_manifest = select_training_run(
                root, values, model, task, base_checkpoint
            )
            adapted_checkpoint = checkpoint_path(training_run, model)
            if not adapted_checkpoint.exists():
                raise FileNotFoundError(
                    f"Fine-tuned checkpoint missing for {model} {task.key}: "
                    f"{adapted_checkpoint}"
                )
            training = training_manifest["artifact_metadata"]["training"]
            for state, model_path, adaptation in (
                (
                    "frozen",
                    base_checkpoint,
                    {"state": "frozen", "source_checkpoint": str(base_checkpoint)},
                ),
                (
                    "task_finetuned",
                    adapted_checkpoint,
                    {
                        "state": "task_finetuned",
                        "training_manifest": str(training_run / "manifest.json"),
                        "source_checkpoint": training_manifest["runtime_config"]["source_checkpoint"],
                        "seed": training_manifest["pipeline_config"]["seed"],
                        "training_config": training_manifest["model_config"]["training"],
                        "training_implementation": training["implementation"],
                    },
                ),
            ):
                print(f"[{index}/{len(tasks)}] evaluate {model} {state} on {task.key}")
                _evaluate(
                    model,
                    task.dataset,
                    task.term,
                    model_path,
                    root / "tasks" / state,
                    config_path,
                    int(model_cfg["context_length"]),
                    int(values["evaluation"]["batch_size"]),
                    adaptation,
                )
    finally:
        if previous_experiment is None:
            os.environ.pop("TIME_EXPERIMENT", None)
        else:
            os.environ["TIME_EXPERIMENT"] = previous_experiment


def _evaluate(
    model: str,
    dataset: str,
    term: str,
    model_path: Path,
    output_dir: Path,
    config_path: Path | None,
    context_length: int,
    batch_size: int,
    adaptation: dict,
) -> None:
    common = dict(
        dataset_name=dataset,
        terms=[term],
        output_dir=str(output_dir),
        batch_size=batch_size,
        context_length=context_length,
        config_path=config_path,
        model_path=model_path,
        covariate_mode="none",
        target_mode="univariate",
        instance_normalization="none",
        adaptation=adaptation,
    )
    if model == "chronos2":
        run_chronos2_experiment(model_size="chronos2", **common)
    elif model == "chronos_bolt":
        run_chronos_bolt_experiment(model_size="base", **common)
    else:
        run_tsicl_experiment(model_size="tsicl-v1", **common)


if __name__ == "__main__":
    main()
