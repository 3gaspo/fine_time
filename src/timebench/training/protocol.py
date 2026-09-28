"""Shared data and artifact contract for per-task foundation-model fine-tuning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
from timebench.paths import outputs_root
from timebench.pipeline.runs import RunHandle, allocate_run, select_completed_runs


FINE_TIME_MODELS = ("chronos_bolt", "chronos2", "ts_icl")
SELECTIME_EXCLUDED_DATASETS = (
    "Coastal_T_S/5T",
    "current_velocity/20T",
    "azure2019_D/5T",
    "azure2019_I/5T",
)


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    term: str
    prediction_length: int
    test_length: int
    val_length: int

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.term}"


def select_tasks(
    config_path: str | Path | None,
    excluded_datasets: Iterable[str] = SELECTIME_EXCLUDED_DATASETS,
    dataset: str | None = None,
    term: str | None = None,
    expected_count: int | None = None,
) -> list[TaskSpec]:
    """Return the configured TIME tasks after applying the Selectime exclusions."""

    config = load_dataset_config(Path(config_path) if config_path else None)
    excluded = set(excluded_datasets)
    tasks: list[TaskSpec] = []
    for dataset_name, dataset_config in config.get("datasets", {}).items():
        if dataset_name in excluded or (dataset is not None and dataset_name != dataset):
            continue
        available_terms = [
            name
            for name, value in dataset_config.items()
            if name in {"short", "medium", "long"} and isinstance(value, Mapping)
        ]
        for task_term in available_terms:
            if term is not None and task_term != term:
                continue
            settings = get_dataset_settings(dataset_name, task_term, config)
            tasks.append(
                TaskSpec(
                    dataset=dataset_name,
                    term=task_term,
                    prediction_length=int(settings["prediction_length"]),
                    test_length=int(settings["test_length"]),
                    val_length=int(settings["val_length"] or 0),
                )
            )
    tasks.sort(key=lambda item: (item.dataset, item.term))
    if not tasks:
        raise ValueError("The selected Fine TIME task set is empty")
    if dataset is None and term is None and expected_count is not None and len(tasks) != expected_count:
        raise ValueError(
            f"Fine TIME expected {expected_count} tasks after exclusions, found {len(tasks)}"
        )
    return tasks


def load_training_series(task: TaskSpec, min_context: int) -> list[np.ndarray]:
    """Load one task's official training prefix as independent univariate series."""

    dataset = Dataset(
        name=task.dataset,
        term=task.term,
        to_univariate=False,
        prediction_length=task.prediction_length,
        test_length=task.test_length,
        val_length=task.val_length,
    )
    if dataset.target_dim > 1:
        dataset = Dataset(
            name=task.dataset,
            term=task.term,
            to_univariate=True,
            prediction_length=task.prediction_length,
            test_length=task.test_length,
            val_length=task.val_length,
        )
    series: list[np.ndarray] = []
    for entry in dataset.training_dataset:
        values = np.asarray(entry["target"], dtype=np.float32).reshape(-1)
        if np.isinf(values).any():
            continue
        if _has_usable_window(values, min_context, task.prediction_length):
            series.append(values)
    if not series:
        raise ValueError(
            f"{task.key} has no model-eligible training series with finite "
            "input and output support"
        )
    return series


def _has_usable_window(values: np.ndarray, min_context: int, horizon: int) -> bool:
    if len(values) < min_context + horizon:
        return False
    for cut in range(min_context, len(values) - horizon + 1):
        if np.isfinite(values[max(0, cut - min_context) : cut]).any() and np.isfinite(
            values[cut : cut + horizon]
        ).any():
            return True
    return False


def sample_window_batch(
    series: Sequence[np.ndarray],
    *,
    batch_size: int,
    context_length: int,
    min_context: int,
    horizon: int,
    generator: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample deterministic random train windows and left-pad their contexts."""

    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(1000, 100 * batch_size)
    while len(contexts) < batch_size and attempts < max_attempts:
        attempts += 1
        values = series[int(generator.integers(len(series)))]
        maximum_cut = len(values) - horizon
        if maximum_cut < min_context:
            continue
        cut = int(generator.integers(min_context, maximum_cut + 1))
        context = values[max(0, cut - context_length) : cut]
        target = values[cut : cut + horizon]
        if np.isinf(context).any() or np.isinf(target).any():
            continue
        if not np.isfinite(context).any() or not np.isfinite(target).any():
            continue
        padded = np.full(context_length, np.nan, dtype=np.float32)
        padded[-len(context) :] = context
        contexts.append(padded)
        targets.append(target.astype(np.float32, copy=False))
    if len(contexts) != batch_size:
        raise ValueError("Could not sample a complete model-eligible training batch")
    return torch.from_numpy(np.stack(contexts)), torch.from_numpy(np.stack(targets))


def set_seed(seed: int) -> np.random.Generator:
    """Set the run's sole seed on every stochastic library used here."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def output_root(configured: str | Path | None = None) -> Path:
    return Path(configured).expanduser().resolve() if configured else outputs_root() / "task_finetuning"


def training_identity_root(root: Path, model: str, task: TaskSpec) -> Path:
    return root / "checkpoints" / model / task.dataset / task.term


def checkpoint_path(run_dir: Path, model: str) -> Path:
    return run_dir / ("finetuned.ckpt" if model == "ts_icl" else "finetuned-ckpt")


def model_training_config(config: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    models = config["training"]["models"]
    if model not in models:
        raise ValueError(f"No training configuration for {model!r}")
    return models[model]


def training_config_snapshot(
    config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(model_config),
        "mode": config["training"]["mode"],
        "min_context": int(config["training"]["min_context"]),
        "gradient_clip_norm": float(config["training"]["gradient_clip_norm"]),
    }


def training_run_config(
    config: Mapping[str, Any],
    model: str,
    task: TaskSpec,
    source_checkpoint: Path,
) -> dict[str, dict[str, Any]]:
    model_config = model_training_config(config, model)
    return {
        "identity": {
            "model": model,
            "dataset": task.dataset,
            "term": task.term,
            "stage": "fine_tuning",
        },
        "model_config": {
            "alias": model,
            "training": training_config_snapshot(config, model_config),
        },
        "pipeline_config": {
            "seed": int(config["seed"]),
            "prediction_length": task.prediction_length,
            "test_length": task.test_length,
            "validation_length": task.val_length,
            "training_split": "TIME prefix before validation and test",
        },
        "runtime_config": {"source_checkpoint": str(source_checkpoint)},
        "experiment_config": {"target_mode": "univariate"},
    }


def allocate_training_run(
    root: Path,
    config: Mapping[str, Any],
    model: str,
    task: TaskSpec,
    source_checkpoint: Path,
) -> RunHandle:
    values = training_run_config(config, model, task, source_checkpoint)
    return allocate_run(
        training_identity_root(root, model, task),
        experiment="task_finetuning",
        provenance={
            "dataset_config": str(config.get("config_path") or "project default"),
        },
        **values,
    )


def select_training_run(
    root: Path,
    config: Mapping[str, Any],
    model: str,
    task: TaskSpec,
    source_checkpoint: Path,
) -> tuple[Path, dict[str, Any]]:
    values = training_run_config(config, model, task, source_checkpoint)
    filters = {
        "experiment": "task_finetuning",
        **{key: value for key, value in values.items()},
    }
    selected = select_completed_runs(
        training_identity_root(root, model, task),
        config_filters=filters,
        config_policy="latest",
        repeat_policy="latest",
    )
    if len(selected) != 1:
        raise FileNotFoundError(
            f"Expected one completed fine-tuning run for {model} {task.key}, "
            f"found {len(selected)}"
        )
    return selected[0]


def checkpoint_artifacts(run_dir: Path, checkpoint: Path) -> list[str]:
    paths = [checkpoint] if checkpoint.is_file() else [
        path for path in checkpoint.rglob("*") if path.is_file()
    ]
    if not paths:
        raise FileNotFoundError(f"Fine-tuned checkpoint is empty: {checkpoint}")
    return [str(path.relative_to(run_dir)) for path in sorted(paths)]
