"""Dependency-free contract checks for the Fine TIME LoRA workflow."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FineTimeLoraContractTest(unittest.TestCase):
    def test_configuration_launchers_and_public_contract_align(self) -> None:
        base = (ROOT / "src/timebench/conf/fine_time.yaml").read_text()
        variant = (ROOT / "src/timebench/conf/fine_time_lora.yaml").read_text()
        launcher = (ROOT / "scripts/fine_time_lora.sh").read_text()
        pyproject = (ROOT / "pyproject.toml").read_text()
        readme = (ROOT / "README.md").read_text()

        for fragment in ("r: 8", "lora_alpha: 16", "lora_dropout: 0.0"):
            self.assertIn(fragment, base)
        for fragment in ("- fine_time", "task_finetuning_lora", "mode: lora"):
            self.assertIn(fragment, variant)
        self.assertIn("for model in chronos2 chronos_bolt ts_icl", launcher)
        for model in ("chronos2", "chronos_bolt", "ts_icl"):
            front = ROOT / f"slurm/selena/fine_time_lora/{model}_selena.slurm"
            text = front.read_text()
            self.assertIn("TIME_FINE_TIME_VARIANT=lora", text)
            self.assertIn(f"TIME_MODEL={model}", text)
        self.assertIn("peft>=0.18.1,<1", pyproject)
        self.assertIn("bash scripts/fine_time_lora.sh selena", readme)

    def test_adapter_helper_is_shared_and_merged(self) -> None:
        helper = (ROOT / "src/timebench/training/lora.py").read_text()
        self.assertIn("LoraConfig", helper)
        self.assertIn("get_peft_model", helper)
        self.assertIn("merge_and_unload", helper)
        for name in ("chronos_bolt", "ts_icl"):
            script = (ROOT / f"src/scripts/finetune_{name}.py").read_text()
            self.assertIn('["training"]["mode"]', script)
            self.assertIn("apply_lora", script)
        chronos2 = (ROOT / "src/scripts/finetune_chronos2.py").read_text()
        self.assertIn("lora_config_dict", chronos2)
        self.assertIn('values["training"]["mode"]', chronos2)


if __name__ == "__main__":
    unittest.main()
