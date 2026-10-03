# Results recap

The first Fine TIME experiment completed the planned 90-task comparison for
`chronos2`, `chronos_bolt`, and `ts_icl`. Every backbone has 90 finite paired
frozen and task-fine-tuned MASE values. Lower MASE is better.

| Model | Frozen mean task MASE | Fine-tuned mean task MASE | Relative change | Task win rate |
| --- | ---: | ---: | ---: | ---: |
| `chronos2` | 1.06923 | 1.05234 | -1.58% | 75.56% |
| `chronos_bolt` | 1.16790 | 1.12307 | -3.84% | 80.00% |
| `ts_icl` | 1.09871 | 1.09506 | -0.33% | 81.11% |

Task-specific fine-tuning improved the equal-task-weighted mean for all three
backbones. Chronos-Bolt obtained the largest aggregate reduction, while
TS-ICL improved on the largest fraction of tasks but by the smallest aggregate
amount. The effect is not uniform: median task-relative changes were -0.93%,
-3.32%, and -0.26%, respectively, and Chronos-Bolt showed the widest spread,
including both the largest gains and the largest degradations.

These results establish the direction and magnitude of the first configured
comparison only. They use one seed, 100 adaptation steps, the selected 90-task
subset, and equal task weights. They do not establish statistical significance,
benefits beyond this profile, or a uniformly safer forecast for every task.
