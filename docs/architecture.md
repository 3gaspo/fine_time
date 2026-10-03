# Architecture

Fine TIME adds a per-task training branch to the inherited Evaluating TSFMs
evaluation path. `src/timebench/training/` owns the 90-task selection, official
TIME training-prefix loading, deterministic window sampling, checkpoint paths,
and training provenance. The three `src/scripts/finetune_*.py` entry points own
model-specific adaptation. `src/scripts/evaluate_fine_time.py` then calls the
unchanged inherited inference adapters for both frozen and adapted states.

`fine_time.yaml` is the full-tuning base contract;
`fine_time_lora.yaml` changes only the output root, experiment name and
training mode. `training/lora.py` attaches the declared PEFT adapters and
merges them after training so the same checkpoint readers evaluate full and
LoRA adaptations. The two modes retain independent manifests and reports.

```text
TIME training prefix --> model-specific fine-tuner --> per-task checkpoint
          |                                           |
          `--------------- frozen checkpoint          |
                              |                       |
                              v                       v
                         inherited test evaluation on one Seasonal grid
                                          |
                                          v
                    paired task MASE table and frozen/adapted scatter
```

`src/timebench/results/fine_tuning.py` requires an exact task pair within each
backbone and gives every TIME task equal weight. The public launcher submits
one sequential 90-task job per backbone, then an `afterok` report job.

The parent layer inherits dataset, model-adapter, covariate, metric, timing,
feature, task-manifest, Seasonal Naive, diagnostic, reporting, artifact, and
cluster-runtime behavior from Improved TIME. The `experiments/` Python entry
points call those common owners. Public commands in `scripts/` select an
experiment family and submit the inherited cluster fronts in `slurm/`, whose
workflow implementations live in `src/slurm/`. The project-owned
`foundation_model_schedule.sh` excludes TimesFM-3 without changing the shared
capability registry.

```text
saved-Arrow TIME dataset
        |
        v
src/timebench/evaluation + experiments/<model>.py
        |
        v
schema-1 task runs in outputs/<experiment>/tasks
        |
        +<-- reusable Seasonal Naive task store
        |
        +--> summary tables
        `--> feature-performance analysis
```

`src/timebench/pipeline/` owns run allocation, exact-configuration identity,
recovery, interruption, and result selection. `src/timebench/feature/` owns
dataset features and reusable association calculations. Experiment scripts
compose these components but do not redefine their contracts.

`src/timebench/results/performance.py` builds generic tables and report
bundles from already selected, repeat/configuration-reduced task statistics.
`src/timebench/visualization/performance.py` plots those aggregates without
reloading models or pooling metric cells. The compact-summary CLI composes
both owners and records every produced artifact in its report manifest.
`src/timebench/pipeline/runtime_resources.py` provides the compute-node
device/memory snapshot invoked once per allocation by the runtime shell.

The inherited Seasonal Naive producer can own a reusable shared task store or
a project-owned task store outside the learned-model and channel roots. Each
task also writes the common evaluation
grid: finite target steps and cells whose Seasonal Naive median and MASE are
finite. Consumers resolve that selected grid before inference, use it for
every metric, and reject non-finite forecasts on its support. Learned-model
jobs never write to the shared store, and summaries wait only for the learned
jobs belonging to their launch.

DGX and Selena retain independent environments and project-scoped output/log
roots. Code synchronization excludes every dataset, weight, output, log,
environment, and private lifecycle file. Results move only between execution
surfaces of this repository.
