"""Focused scientific and artifact contracts for foundation ablations."""

from pathlib import Path
import sys
import types
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DOTENV = types.ModuleType("dotenv")
DOTENV.load_dotenv = lambda *_args, **_kwargs: False
sys.modules.setdefault("dotenv", DOTENV)

from timebench.evaluation.normalization import normalize_instance
from timebench.paths import foundation_experiment_axis, foundation_identity_root


class FoundationAblationContractTest(unittest.TestCase):
    def test_zscore_is_per_variate_and_exactly_invertible(self) -> None:
        context = np.asarray([[1.0, np.nan, 3.0], [5.0, 5.0, np.nan]])
        normalized, owner = normalize_instance(context, "zscore")

        np.testing.assert_array_equal(np.isnan(normalized), np.isnan(context))
        np.testing.assert_allclose(np.nanmean(normalized[0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.nanstd(normalized[0], ddof=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(normalized[1, :2], 0.0, atol=1e-12)
        quantiles = np.stack([normalized, normalized + 1.0])
        restored = owner.inverse_quantiles(quantiles)
        np.testing.assert_allclose(restored[0], context, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(restored[1, 1, 0], 6.0, atol=1e-12)

    def test_experiment_axes_isolate_settings_before_run_n(self) -> None:
        context_axis = foundation_experiment_axis(
            "context_size", context_length=1024, instance_normalization="none"
        )
        context_root = foundation_identity_root(
            Path("outputs/context_size/tasks"),
            "chronos2",
            "univariate",
            "SG_Weather/D",
            "short",
            experiment_axis=context_axis,
        )
        self.assertEqual(
            context_root,
            Path("outputs/context_size/tasks/chronos2/context_length/1024/")
            / "univariate/SG_Weather/D/short",
        )

        normalization_axis = foundation_experiment_axis(
            "instance_normalization",
            context_length=8192,
            instance_normalization="zscore",
        )
        normalization_root = foundation_identity_root(
            Path("outputs/instance_normalization/tasks"),
            "chronos2",
            "univariate",
            "SG_Weather/D",
            "short",
            experiment_axis=normalization_axis,
        )
        self.assertEqual(
            normalization_root,
            Path("outputs/instance_normalization/tasks/chronos2/normalization/zscore/")
            / "univariate/SG_Weather/D/short",
        )


if __name__ == "__main__":
    unittest.main()
