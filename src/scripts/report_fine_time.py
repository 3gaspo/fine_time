#!/usr/bin/env python3
"""Build the Fine TIME paired comparison report."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.results.fine_tuning import build_fine_tuning_report
from timebench.training import FINE_TIME_MODELS, output_root, select_tasks


@hydra.main(version_base=None, config_path="../timebench/conf", config_name="fine_time")
def main(cfg: DictConfig) -> None:
    values = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(values, dict)
    root = output_root(values.get("output_root"))
    tasks = select_tasks(
        values.get("config_path"),
        values["experiment"]["excluded_datasets"],
        expected_count=int(values["experiment"]["expected_tasks"]),
    )
    expected = {task.key for task in tasks} if values["evaluation"]["require_complete"] else None
    artifacts = build_fine_tuning_report(
        root / "tasks",
        root / "reports" / "frozen_vs_task_finetuned",
        FINE_TIME_MODELS,
        expected,
    )
    print("Fine TIME report artifacts:")
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
