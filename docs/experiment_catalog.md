# Experiment catalog

## Fine TIME experiment

The active experiment independently fine-tunes `chronos2`, `chronos_bolt`, and
`ts_icl` on each task's TIME training prefix, then compares each checkpoint
with the same model frozen. It uses univariate targets and the exact 90-task
Selectime subset. The pilot protocol uses 100 full-model optimizer steps and
one seed; these settings are configuration, not validated scientific optima.

Chronos-2 uses its official `fit` API. Chronos-Bolt and TS-ICL use explicit
source-adapted quantile-loss paths because their installed packages do not
provide equivalent fine-tuning commands. `scripts/fine_time.sh` launches the
model jobs and the paired report.

## LoRA comparison

`scripts/fine_time_lora.sh` runs the same three backbones, 90 tasks, seed,
100-step budget and paired evaluation with parameter-efficient adapters. The
fixed comparison uses rank 8, alpha 16, zero dropout, no bias and explicit
target-module lists per backbone. Its artifacts live below
`task_finetuning_lora`, separate from full tuning. The execution environment
must be relocked with the declared PEFT dependency before first submission.

## Inherited runnable families

### Foundation-model benchmark

Compares `chronos_bolt`, `chronos2`, `ts_icl`, and deterministic
`seasonal_naive` over the official TIME test tasks. TimesFM-3 is excluded from
the active experiment set. The benchmark records actual target
mode, scaled MASE, finite/grid/total coverage, and inference seconds. Seasonal
Naive defines one reusable evaluation grid from finite ground-truth support,
finite Seasonal predictions, and finite Seasonal MASE. Every learned model is
evaluated on that grid, and a non-finite forecast on expected support fails the
task. Generate the baseline once with `scripts/submit_seasonal_naive.sh`; then
launch the parallel learned-model jobs with
`scripts/submit_foundation_models.sh`.

### Chronos-2 channel comparison

Compares native multivariate targets, independent univariate targets, and
past targets represented as past-only covariates on multivariate datasets.
Entry point: `scripts/channels_comparison.sh`. Its summaries consume the same
reusable Seasonal Naive baseline as the foundation-model benchmark.

### Maximum-context ablation

Runs `chronos_bolt`, `chronos2`, and `ts_icl` at five maximum contexts, with
each value half the preceding one. The grids are respectively
`2048/1024/512/256/128`, `8192/4096/2048/1024/512`, and
`4096/2048/1024/512/256`. Entry point: `scripts/context_size.sh`. Each setting
has its own `context_length/<value>/` task subtree. The launch summary reports
the ordinary comparison bundle and a horizon-size by context-size scaled-MASE
figure with one panel per model.

### Instance-normalization ablation

Compares unchanged inputs with per-window, per-variate z-score normalization
for `chronos_bolt`, `chronos2`, and `ts_icl`. The finite-value population mean
and standard deviation are computed after maximum-context truncation; original
missing positions remain missing and a zero finite standard deviation uses
scale one. Every output quantile is transformed back before evaluation. Entry
point: `scripts/instance_normalization.sh`. The two settings have separate
`normalization/none/` and `normalization/zscore/` task subtrees.

### Dataset diagnostics

Audits source non-finiteness and forecast windows, then extracts reusable
dataset features. Entry point: `scripts/dataset_diagnostics.sh`.

## Planned families

- covariate ablations across every foundation model that declares support;

Its exact grid, supported-model subset, and aggregation policy have not yet
been selected. The migrated launchers do not silently implement it.
