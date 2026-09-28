"""Focused reporting/packaging check; no models, scheduler or real results."""

import ast
import json
import runpy
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from timebench.results.performance import (
    prepare_tasks, build_performance_tables, build_horizon_frequency_tables,
    write_performance_report,
)

ROOT = Path(__file__).resolve().parents[2]


def tasks():
    rows = []
    for model, losses in [("seasonal_naive", [1, 2, 1, 4]), ("candidate", [.5, 1, 1, 8])]:
        for index, ((dataset, frequency, term, horizon), loss) in enumerate(zip([
            ("Australia_Solar", "D", "short", 4),
            ("Australia_Solar", "D", "long", 8),
            ("Smart_Manufacturing", "H", "short", 4),
            ("Smart_Manufacturing", "H", "long", 8),
        ], losses)):
            rows.append(dict(model=model, dataset=dataset, frequency=frequency, term=term,
                horizon_steps=horizon, MASE=loss, MASE_std=loss / 2, MASE_variance=loss ** 2 / 4,
                seasonal_MASE_variance=[1, 2, 1, 4][index] ** 2 / 4,
                scaled_MASE=loss / [1, 2, 1, 4][index],
                inference_seconds=None if model == "candidate" and index == 3 else 1.0))
    return rows


class ReportingBundleTest(unittest.TestCase):
    def test_source_and_packaging_contract(self):
        for directory in ("src", "scripts", "experiments"):
            for path in (ROOT / directory).rglob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        domains = json.loads((ROOT / "src/timebench/config/dataset_domains.json").read_text())
        self.assertEqual(domains["Australia_Solar"], "Climate")
        self.assertIn("!/src/timebench/config/dataset_domains.json", (ROOT / ".gitignore").read_text())
        reporter = (ROOT / "scripts/compute_foundation_summary.py").read_text()
        self.assertIn("performance_artifacts = write_performance_artifacts(", reporter)
        self.assertIn("artifacts=[args.csv, args.markdown, *performance_artifacts]", reporter)
        for retired in ("scripts/compute_local_leaderboard.py", "scripts/run_all_foundation_models.sh",
                        "src/slurm/benchmark_foundation_models.sh",
                        "src/scripts/backfill_seasonal_dispersion.py"):
            self.assertFalse((ROOT / retired).exists(), retired)

    def test_aggregation_pairing_ties_and_missing_latency(self):
        values, summary, domain = build_performance_tables(
            prepare_tasks(tasks()), reference="seasonal_naive")
        row = summary.set_index("model").loc["candidate"]
        self.assertAlmostEqual(row.scaled_MASE, .5 ** .25)
        self.assertAlmostEqual(row.relative_improvement_percent, -31.25)
        self.assertAlmostEqual(row.mean_paired_improvement_percent, 0)
        self.assertTrue(pd.isna(row.total_inference_seconds))
        self.assertEqual(row.timed_tasks, 3)
        _, best = build_horizon_frequency_tables(values)
        tied = best[(best.frequency == "H") & (best.horizon_steps == 4)].iloc[0]
        self.assertEqual(set(json.loads(tied.winners)), {"seasonal_naive", "candidate"})
        unmatched = values.drop(values[(values.model == "candidate") & (values.frequency == "H")].index)
        _, best = build_horizon_frequency_tables(unmatched)
        self.assertFalse(best.loc[best.frequency == "H", "comparable"].any())
        self.assertEqual(set(domain.domain), {"Climate", "Industry"})

    def test_bundle_exports_and_transfer_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = write_performance_report(tasks(), root, reference="seasonal_naive")
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in artifacts))
            manifest = json.loads((root / "performance_report_manifest.json").read_text())
            self.assertEqual(manifest["scaled_MASE_aggregation"], "geometric")
            self.assertEqual(set(manifest["artifacts"]),
                             {path.name for path in artifacts if path.name != "performance_report_manifest.json"})
            for name in ("performance_summary.tex", "relative_improvement.md", "time_totals.csv",
                         "domain_average_loss.csv", "loss_horizon_frequency.pdf",
                         "accuracy_time.pdf", "task_mean_std.png", "task_mean_std.pdf",
                         "task_dispersion.png", "task_dispersion.pdf",
                         "scaled_MASE_horizon_frequency.png", "best_model_relative_loss_horizon_frequency.pdf"):
                self.assertTrue((root / name).exists(), name)
            self.assertEqual(manifest["dispersion_available_tasks"], {"seasonal_naive": 4, "candidate": 4})
            for png in root.glob("*.png"):
                self.assertTrue(png.with_suffix(".pdf").is_file(), png.name)
        for script in ("sync_results_to_dgx.sh", "publish_job.sh"):
            self.assertIn("src/timebench/pipeline/artifact_selection.py", (ROOT / script).read_text())

    def test_lightweight_selection(self):
        selector = runpy.run_path(str(ROOT / "src/timebench/pipeline/artifact_selection.py"))
        selected = selector["selected"]
        for name in ("channels_comparison/reports/native/performance/task_mean_std.png",
                     "foundation_models/reports/performance/accuracy_time.pdf",
                     "tasks/cell/prediction.json", "tasks/cell/time_inference/record.json"):
            self.assertTrue(selected(name, "lightweight"), name)
        for name in ("foundation_models/reports/raw.pt", "tasks/cell/metrics.npz", "tasks/cell/predictions.npy"):
            self.assertFalse(selected(name, "lightweight"), name)
        self.assertTrue(selected("tasks/cell/metrics.npz", "detailed"))
        self.assertIn("--exclude=*.npz", selector["filters"]("lightweight"))
        transfer = (ROOT / "sync_results_to_dgx.sh").read_text()
        publisher = (ROOT / "publish_job.sh").read_text()
        self.assertIn('PUBLISH_MAX_FILE_BYTES:-100000000', transfer)
        self.assertIn('PUBLISH_MAX_FILE_BYTES:-100000000', publisher)
        workflow = (ROOT / "src/slurm/run_chronos2_comparison.sh").read_text()
        self.assertIn('TIME_WORKFLOW_NAME=channels_summary', workflow)
        self.assertIn('if [ "${TIME_REPORT_ONLY:-0}" != 1 ]; then', workflow)


if __name__ == "__main__":
    unittest.main()
