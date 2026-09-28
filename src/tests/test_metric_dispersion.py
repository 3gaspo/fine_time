"""Dependency-light regression for the maintained model-job statistics."""

from pathlib import Path
import unittest

import numpy as np

from timebench.evaluation.metrics import summarize_metric_values


class MetricDispersionTest(unittest.TestCase):
    def test_population_statistics_and_empty_support(self):
        values = np.array([[[1.0, np.nan], [3.0, np.inf]]])
        result = summarize_metric_values(values, 3)
        self.assertEqual(result, {
            "mean": 2.0, "std": 1.0, "variance": 1.0, "dispersion_ddof": 0,
            "finite_values": 2, "evaluation_values": 3, "total_values": 4,
        })
        empty = summarize_metric_values(np.array([np.nan]), 0)
        for field in ("mean", "std", "variance"):
            self.assertIsNone(empty[field])
        self.assertEqual(summarize_metric_values(np.array([7.0]), 1)["std"], 0.0)

    def test_mean_is_unchanged_and_saver_uses_shared_statistics(self):
        values = np.array([1.0, 2.0, 7.0, np.nan], dtype=np.float32)
        result = summarize_metric_values(values, 3)
        self.assertEqual(result["mean"], float(np.mean(values[np.isfinite(values)])))
        self.assertAlmostEqual(result["std"] ** 2, result["variance"])
        saver = Path(__file__).parents[1] / "timebench/evaluation/saver.py"
        self.assertIn("metric_summaries[metric_name] = summarize_metric_values(", saver.read_text(encoding="utf-8"))
        seasonal = Path(__file__).parents[2] / "experiments/seasonal_naive.py"
        self.assertIn("create_evaluation_grid=True", seasonal.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
