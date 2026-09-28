# Results recap

## Fine TIME evidence status

No Fine TIME fine-tuning run has been completed or analyzed yet. The results
below are inherited frozen-model evidence from `evaluating_tsfms`; they justify
the selected backbones and test protocol but are not evidence that per-task
fine-tuning helps.

The [scientific executive summary](../latex/executive_summary.pdf)
([LaTeX source](../latex/executive_summary.tex)) gives the mathematical task,
essential protocols, current-equivalent findings, and supporting tables and
plots. The recap below is the concise evidence overview.

## Completed shared-grid foundation benchmark

The current Evaluating TSFMs artifact lineage covers 50 TIME dataset-frequency
configurations and 98 horizon tasks for each active foundation model. Scaled
MASE divides each task MASE by its matching Seasonal Naive MASE and takes the
geometric mean over tasks; lower is better. Inference seconds sum recorded
forecast-loop wall times, including loop preparation but excluding model
loading, metric computation, and artifact saving. TimesFM-3 is intentionally
excluded from the active experiment set.

| Model | Scaled MASE | Dataset-frequency IQR | Configurations below baseline | Learned-model task wins | Inference seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos-2 | 0.685489 | 0.612–0.774 | 48/50 | 74/98 | 439.1 |
| TS-ICL | 0.718394 | 0.673–0.808 | 47/50 | 20/98 | 2,672.5 |
| Chronos-Bolt | 0.759559 | 0.694–0.842 | 47/50 | 4/98 | 942.1 |
| Seasonal Naive | 1.000000 | 1.000–1.000 | — | — | 654.6 |

Chronos-2 is the strongest active foundation model. Its aggregate scaled MASE
is 31.5% below Seasonal Naive, 4.6% below TS-ICL, and 9.8% below Chronos-Bolt.
Its advantage is broad rather than driven only by the geometric aggregate: it
wins 74 of the 98 paired learned-model tasks. TS-ICL and Chronos-Bolt require
6.09 and 2.15 times Chronos-2's summed inference time, respectively.

Every active learned model completed all 98 selected task manifests and is
finite in MASE on all 111,071 series-window-variate metric cells in the shared
Seasonal grid. The grid excludes 106 of the 111,177 candidate metric cells
because their target or Seasonal baseline support is not finite. These counts
are not individual horizon timesteps. Identical support makes the model
ranking directly comparable.

## Context-size study

The completed context-size launch evaluates five model-specific maximum input
lengths on the same 98 tasks and 111,071-cell Seasonal grid for every setting.
All 1,470 model-setting task manifests and the aggregate report completed.

| Model | Context | Scaled MASE | Inference seconds | Accuracy versus maximum context | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos-2 | 8,192 | 0.685489 | 425.8 | reference | — |
| Chronos-2 | 4,096 | **0.682427** | 252.5 | 0.45% better | 40.7% |
| Chronos-2 | 2,048 | 0.683788 | 153.5 | 0.25% better | 63.9% |
| Chronos-Bolt | 2,048 | **0.759559** | 957.2 | reference | — |
| Chronos-Bolt | 1,024 | 0.769732 | 585.0 | 1.34% worse | 38.9% |
| TS-ICL | 4,096 | **0.718394** | 2,659.5 | reference | — |
| TS-ICL | 2,048 | 0.719896 | 1,076.3 | 0.21% worse | 59.5% |

Chronos-2 does not benefit monotonically from the longest context: 4,096 has
the best aggregate loss, while 2,048 retains nearly the same accuracy at much
lower recorded cost. The 4,096-versus-8,192 task split is 49 improvements, 28
exact ties, and 21 degradations, with a median task change of only -0.01%, so
the aggregate advantage is small and heterogeneous. Chronos-Bolt and TS-ICL
are most accurate at their longest tested contexts, but halving context offers
a substantial time saving for modest aggregate loss. Further truncation is
progressively harmful: the shortest settings are 12.97% worse for Bolt and
15.59% worse for TS-ICL than their maxima. Population MASE standard deviation
and variance are present for every task/setting; they describe within-task
cell dispersion, not repeated-run uncertainty.

Recorded seconds are forecast-loop totals from one launch, not end-to-end job
wall time. The context-horizon report is readable, but the 15-series
accuracy/time figure has overlapping long labels and should be relabeled before
presentation use.

## Instance-normalization study remains incomplete

The unchanged-input arms completed all 98 tasks for each model and reproduce
the corresponding maximum-context MASE means, standard deviations, variances,
and finite support exactly. Their timing totals differ by at most 2.1% between
launches, consistent with launch-to-launch timing variation.

The z-score arms are not result evidence. Each model completed six tasks and
then failed on `current_velocity/10T/short`: Chronos-Bolt and Chronos-2 produced
non-finite forecasts, while TS-ICL rejected an all-NaN normalized sample. The
failed launch used ordinary mean and population standard deviation on input
contexts containing missing values, making the fitted transform NaN and
defeating the models' successful unchanged-input missing-value handling. The
implementation now computes statistics over finite context values while
preserving original missing positions. No aggregate normalization report was
produced; the z-score arms and dependent summary must be recovered before the
normalization hypothesis can be evaluated.

## Within-task MASE dispersion

The transferred summaries include population variance and standard deviation
for all 294 selected foundation tasks (98 per model). Both use the same finite
series-window-variate metric cells as each task's arithmetic mean, with
`ddof=0`; prior means, coverage, and metadata were verified unchanged.

The executive summary includes the standardized raw task mean-versus-population-
standard-deviation scatter and a paired Seasonal-relative variance scatter,
both with one color per model. The spread concerns evaluated cells within
tasks, not uncertainty across repeated runs. The reproducible plot entry point
is `src/visualization/plot_executive_summary.py`.

## Chronos-2 channel representation

The completed channel comparison covers the 38 multivariate dataset-frequency
configurations and 74 horizon tasks common to its three modes.

| Representation | Scaled MASE | Accuracy versus native | Paired task-ratio IQR | Inference seconds | Time versus native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native multivariate | 0.690045 | reference | 1.000–1.000 | 340.0 | 1.00x |
| Independent univariate | 0.695590 | 0.80% worse | 0.997–1.014 | 350.2 | 1.03x |
| Past targets as covariates | 0.690045 | equivalent | 1.000–1.000 | 1,930.8 | 5.68x |

Native multivariate forecasting wins 46 of the 74 paired tasks against
independent univariate forecasting; univariate wins 28. The aggregate native
advantage is only 0.8%, and the paired univariate/native MASE ratio has an IQR
of 0.997–1.014, so the effect is small and heterogeneous.

Past-target covariates reproduce native-multivariate accuracy to numerical
precision: their aggregate values agree to six decimals and the maximum
taskwise relative MASE difference is below `5.75e-7`. They require 5.68 times
the summed inference time, so this experiment provides no accuracy
justification for the more expensive representation.

All three modes completed 74 selected manifests and are finite on all 91,319
series-window-variate metric cells in their shared grid. Their aggregate reports now complete successfully
against the separately generated Seasonal baseline; the earlier summary-
ordering race is resolved.

## Dataset-feature associations

The completed feature analysis joins dataset-level scaled MASE to 50
dataset-frequency feature rows per learned model. The strongest consistent
Spearman associations are higher temporal heterogeneity (`rho` from +0.42 to
+0.50), higher trend Hurst values (+0.36 to +0.44), higher temporal scale and
location heterogeneity (+0.33 to +0.44), and a negative association with the
second detected period (-0.33 to -0.43). These associations are exploratory,
not causal, and no multiplicity-adjusted significance analysis was performed.
The second detected period has 39 pairwise finite configuration rows per model;
the other displayed associations use all 50. Full-series features include
held-out observations, so the analysis is descriptive rather than a validated
prospective model-selection rule.

## Limitations and next evidence

The Seasonal-paired scatter contains 294 points (98 tasks/model). Its axes
are model/Seasonal task mean MASE and model/Seasonal population MASE variance.
Median variance ratios are 0.590 (Chronos-2), 0.580 (TS-ICL), and 0.629
(Chronos-Bolt); lower variance occurs on 93/98, 91/98, and 84/98 tasks.
Both mean and variance improve on 92/98, 89/98, and 80/98. Variance ratios
above 1 indicate greater dispersion than Seasonal. These are within-task
cell statistics, not repeated-run uncertainty.

- Each scientific configuration has one selected run. Cross-task and cross-
  dataset dispersion does not measure stochastic repeat or seed variability.
- The lightweight artifact snapshot contains terminal logs, manifests,
  configurations, metric summaries, aggregate reports, and feature-analysis
  outputs, but omits task-level `predictions.npz` and `metrics.npz`. Completed
  manifests report those required payloads on Selena, but their contents were
  not independently inspected in this checkout. The current four report
  bundles contain complete task tables and 12 paired PNG/PDF figures each.
- Covariate generalization beyond Chronos-2 remains planned. The context-size
  study is complete; the input-normalization implementation is repaired but
  still has no valid aggregate z-score result until cluster recovery completes.
