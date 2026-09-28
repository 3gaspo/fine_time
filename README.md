# Fine TIME

Fine TIME measures whether task-specific fine-tuning improves time-series
foundation models over their frozen versions on the TIME benchmark. The first
experiment covers `chronos2`, `chronos_bolt`, and `ts_icl`.

## Status

| Implementation | Experiments | Next milestone |
| --- | --- | --- |
| Runnable pilot implemented | None run | Execute the first frozen-versus-fine-tuned comparison for all three backbones |

The training, paired evaluation, caching, and reporting paths are implemented.
No Fine TIME result is available yet.

## First experiment

Each task starts from the same public frozen checkpoint for a backbone. The
model is fine-tuned only on that task's official training prefix, ending before
the validation and test intervals, and is then compared with its frozen
counterpart on identical official test windows.

The first experiment uses the 90-task `small` profile inherited from the
benchmark evaluation. It excludes all terms from `Coastal_T_S/5T`,
`current_velocity/20T`, `azure2019_D/5T`, and `azure2019_I/5T`. Seasonal Naive
defines the common finite evaluation cells and MASE reference but is not a
trainable model.

For every task, the workflow:

1. loads the official training prefix;
2. reloads the common frozen checkpoint and fits an independent task-specific
   model;
3. evaluates frozen and fine-tuned checkpoints on the same univariate test
   windows;
4. records paired metrics and provenance.

The validation interval separates training from test data; it is not used to
select a model configuration. Fine-tuning requires at least one training
window with finite input and output support.

## Adaptation methods

The pilot uses one seed (`2021`), 100 optimizer steps per task, full-model
adaptation, and no covariates. Its configuration is defined in
`src/timebench/conf/fine_time.yaml`.

- Chronos-2 uses the official `Chronos2Pipeline.fit` API.
- Chronos-Bolt uses the official forward pass and native multi-quantile loss
  because its pipeline exposes no equivalent fine-tuning method.
- TS-ICL trains the exposed forecaster with quantile loss because version 0.2.1
  exposes no public training utility.

Chronos-2 trains directly at each task horizon. Chronos-Bolt and TS-ICL train
within their native target-length limits and retain their original rollout
behavior for longer TIME horizons.

## Data and reproducibility

The committed environment and configuration files define the experiment.
TIME's saved-Arrow datasets are read from `TIME_DATASET`; model checkpoints are
read from `TIME_WEIGHTS` using these paths:

- `chronos2/`;
- `chronos-bolt-base/`;
- `tsicl/tsicl-v1.ckpt`.

Reusable raw forecasts are separated from metric reductions. Cache identity
includes whether the checkpoint is frozen or task-fine-tuned, so the two
conditions cannot reuse each other's predictions.

On Selena, start all three backbone jobs and their dependent summary with:

```bash
bash scripts/fine_time.sh selena
```

The corresponding DGX front is `bash scripts/fine_time.sh dgx`.

## Outputs

Fine-tuned checkpoints and their authoritative manifests are written under
`<O>/task_finetuning/checkpoints/<model>/<dataset>/<term>/run_n/`.
Frozen and adapted task artifacts remain separate under
`<O>/task_finetuning/tasks/{frozen,task_finetuned}/`, and the paired report
lives under `<O>/task_finetuning/reports/frozen_vs_task_finetuned/`.
Here `<O>` is `outputs/dgx` for DGX/local execution, Selena's scratch output
root during execution, or `outputs/selena` after synchronization.
The independent shared Seasonal checkout stores its completed cells below
`outputs/seasonal_naive/evaluations/`; Fine TIME resolves them through
`TIME_SEASONAL_EVALUATIONS_ROOT`.

Every Slurm stream, Hydra directory, stage log, and workflow status is grouped
below `logs/<surface>/task_finetuning/`. Launch IDs and timestamps remain in
manifests and log records rather than directory names. Each `run_n/manifest.json`
contains the complete result-changing configuration omitted from the path;
training measurements remain in `training_metrics.json`.

The report pairs task-level MASE values, reports equal-task-weighted means and
win rates, and produces frozen-versus-fine-tuned scatter and paired-summary
figures in both PNG and PDF formats. Lower MASE is better.

## Documentation and source tree

- [Architecture](docs/architecture.md) describes the inherited evaluation path
  and the added training and reporting components.
- [Experiment catalog](docs/experiment_catalog.md) specifies the Fine TIME
  comparison.
- [Results recap](docs/results_recap.md) records the current evidence boundary.
- `experiments/`: inherited frozen-inference entry points.
- `src/scripts/finetune_*.py`: one fine-tuning entry point per backbone.
- `src/scripts/evaluate_fine_time.py`: paired frozen/adapted evaluation.
- `src/timebench/training/`: task selection, windows, paths, and provenance.
- `src/timebench/results/`: paired aggregation and plots.
- `src/tests/`: focused scientific and lifecycle checks.

The inherited TIME code remains under the Apache-2.0 license.
