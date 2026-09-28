"""Dependency-light maintenance checks for Evaluating TSFMs."""

from __future__ import annotations

import ast
import re
import tomllib
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


class EvaluatingTSFMsMaintenanceContractTest(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "experiments", PROJECT_ROOT / "scripts"]
        paths = sorted(path for root in roots for path in root.rglob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_configuration_files_parse(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
            tomllib.load(stream)
        with (PROJECT_ROOT / "src/timebench/config/datasets.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertIn("datasets", config)

    def test_local_document_links_exist(self) -> None:
        markdown = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
        missing: list[str] = []
        for document in markdown:
            for target in LOCAL_LINK.findall(document.read_text(encoding="utf-8")):
                target = target.strip().split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (document.parent / target).resolve()
                if not resolved.exists():
                    missing.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_private_lifecycle_files_are_ignored(self) -> None:
        ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        required = {
            "AGENTS.md",
            "/FUTURE_WORK.md",
            "/PENDING_UPDATES.md",
            "/CLUSTER_STATUS.txt",
            "/docs/INTERNAL_WORKFLOW.md",
        }
        self.assertTrue(required.issubset(set(ignore)))

    def test_split_boundaries_use_declared_intervals(self) -> None:
        source = (PROJECT_ROOT / "src/timebench/evaluation/data.py").read_text(encoding="utf-8")
        self.assertIn("offset=-(self._test_length + self._val_length)", source)
        self.assertIn("offset=-self._test_length", source)
        self.assertIn("math.floor(self._test_length / self.prediction_length)", source)
        self.assertIn("math.floor(self._val_length / self.prediction_length)", source)

    def test_foundation_runners_are_offline_and_fail_fast(self) -> None:
        experiments = {
            "chronos_bolt.py": ("local_files_only=True", "chronos-bolt-{model_size}"),
            "chronos2.py": (
                "BaseChronosPipeline.from_pretrained",
                "local_files_only=True",
                '"chronos2"',
            ),
            "ts_icl.py": ("allow_auto_download=False", '"tsicl/tsicl-v1.ckpt"'),
            "seasonal_naive.py": (),
        }
        for name, required in experiments.items():
            source = (PROJECT_ROOT / "experiments" / name).read_text(encoding="utf-8")
            self.assertNotIn("Failed to run experiment", source, name)
            self.assertNotIn("except Exception", source, name)
            for text in required:
                self.assertIn(text, source, name)

        runtime = (PROJECT_ROOT / "src/slurm/selena_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("export HF_HUB_OFFLINE=1", runtime)
        self.assertIn("export HF_DATASETS_OFFLINE=1", runtime)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", runtime)

    def test_raw_inference_is_separate_from_reductions_and_adaptation(self) -> None:
        runners = (
            "chronos2.py",
            "chronos_bolt.py",
            "ts_icl.py",
            "run_timesfm3.py",
            "seasonal_naive.py",
        )
        for name in runners:
            source = (PROJECT_ROOT / "experiments" / name).read_text(encoding="utf-8")
            self.assertIn('Path(output_dir).parent / "inference"', source, name)
            self.assertIn("save_raw_inference", source, name)
            self.assertIn("load_raw_inference", source, name)
            self.assertIn('"raw_inference": dependency_reference', source, name)
            self.assertNotIn('"val_length": val_length', source, name)
        for name in ("chronos2.py", "chronos_bolt.py", "ts_icl.py"):
            source = (PROJECT_ROOT / "experiments" / name).read_text(encoding="utf-8")
            self.assertIn('"adaptation": adaptation', source, name)
        cache = (PROJECT_ROOT / "src/timebench/pipeline/inference_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"raw_predictions.npz"', cache)
        self.assertIn('"inference.json"', cache)

    def test_shared_seasonal_workflow_contract(self) -> None:
        producer = (PROJECT_ROOT / "scripts/submit_seasonal_naive.sh").read_text(
            encoding="utf-8"
        )
        launcher = (PROJECT_ROOT / "scripts/submit_foundation_models.sh").read_text(
            encoding="utf-8"
        )
        registry = (
            PROJECT_ROOT / "src/slurm/foundation_model_runners.sh"
        ).read_text(encoding="utf-8")
        summary = (
            PROJECT_ROOT / "src/slurm/summarize_foundation_models.sh"
        ).read_text(encoding="utf-8")
        channels = (
            PROJECT_ROOT / "src/slurm/run_chronos2_comparison.sh"
        ).read_text(encoding="utf-8")
        seasonal_experiment = (
            PROJECT_ROOT / "experiments/seasonal_naive.py"
        ).read_text(encoding="utf-8")
        evaluation_grid = (
            PROJECT_ROOT / "src/timebench/evaluation/grid.py"
        ).read_text(encoding="utf-8")
        grid_resolver = (
            PROJECT_ROOT / "src/timebench/pipeline/evaluation_grid.py"
        ).read_text(encoding="utf-8")
        saver = (
            PROJECT_ROOT / "src/timebench/evaluation/saver.py"
        ).read_text(encoding="utf-8")

        self.assertIn("dgx|selena", producer)
        self.assertIn("OUTPUTS_ROOT=$TIME_SEASONAL_OUTPUTS_ROOT", producer)
        self.assertIn('for model in "${FOUNDATION_LEARNED_MODELS[@]}"', launcher)
        self.assertNotIn("seasonal_job", launcher)
        self.assertIn('--dependency="afterany:$dependency"', launcher)
        self.assertIn("FOUNDATION_LEARNED_MODELS=(", registry)
        self.assertIn(
            '--seasonal-naive-results-dir "$TIME_SEASONAL_EVALUATIONS_ROOT"', summary
        )
        self.assertIn(
            '--seasonal-naive-results-dir "$TIME_SEASONAL_EVALUATIONS_ROOT"', channels
        )
        self.assertIn("create_evaluation_grid=True", seasonal_experiment)
        self.assertIn("finite_ground_truth_and_seasonal_naive_mase", evaluation_grid)
        self.assertIn("TIME_SEASONAL_EVALUATIONS_ROOT", grid_resolver)
        self.assertIn("evaluation_grid_path is required", saver)

    def test_seasonal_naive_uses_direct_deterministic_quantiles(self) -> None:
        experiment = (PROJECT_ROOT / "experiments/seasonal_naive.py").read_text(
            encoding="utf-8"
        )
        predictor = (
            PROJECT_ROOT / "src/timebench/models/statsforecast_predictor.py"
        ).read_text(encoding="utf-8")
        for source in (experiment, predictor):
            self.assertNotIn("num_samples", source)
            self.assertNotIn("np.random", source)
        self.assertIn("forecast.quantiles", experiment)
        self.assertIn("self._quantiles = np.stack", predictor)

    def test_time_dataset_download_and_current_model_surface(self) -> None:
        downloader = (PROJECT_ROOT / "scripts/download_time_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('DEFAULT_REPO_ID = "Real-TSF/TIME"', downloader)
        self.assertIn('os.environ["HF_HUB_DISABLE_XET"] = "1"', downloader)
        self.assertLess(
            downloader.index('os.environ["HF_HUB_DISABLE_XET"] = "1"'),
            downloader.index("from huggingface_hub import"),
        )
        self.assertIn("resolved_revision = info.sha", downloader)
        self.assertIn('REVISION_FILE = ".time_snapshot_revision"', downloader)
        self.assertIn("if destination_has_files and not resume", downloader)
        self.assertIn("max_workers=max_workers", downloader)
        self.assertIn('destination.rglob("state.json")', downloader)

        self.assertFalse((PROJECT_ROOT / "experiments/tirex_model.py").exists())
        self.assertFalse((PROJECT_ROOT / "scripts/run_tirex.sh").exists())
        self.assertFalse(
            (PROJECT_ROOT / "slurm/dgx/foundation_models/tirex.slurm").exists()
        )
        self.assertFalse(
            (PROJECT_ROOT / "slurm/selena/foundation_models/tirex_selena.slurm").exists()
        )
        dependencies = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["dependencies"]
        self.assertFalse(any("tirex" in dependency.lower() for dependency in dependencies))
        registry = (PROJECT_ROOT / "src/slurm/foundation_model_runners.sh").read_text(
            encoding="utf-8"
        )
        summary = (PROJECT_ROOT / "scripts/compute_foundation_summary.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("tirex", registry.lower())
        self.assertNotIn('"tirex"', summary.lower())


if __name__ == "__main__":
    unittest.main()
