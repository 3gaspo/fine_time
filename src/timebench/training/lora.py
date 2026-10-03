"""Shared LoRA configuration for Fine TIME foundation-model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn


def lora_config_dict(
    config: Mapping[str, Any],
    model: str,
    target_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the Chronos-2-default LoRA parameters for one backbone."""

    values = config["training"]["lora"]
    targets = target_modules or values["target_modules"][model]
    return {
        "r": int(values["r"]),
        "lora_alpha": int(values["lora_alpha"]),
        "lora_dropout": float(values["lora_dropout"]),
        "bias": str(values["bias"]),
        "target_modules": list(targets),
    }


def apply_lora(
    model: nn.Module,
    config: Mapping[str, Any],
    model_name: str,
    target_modules: Sequence[str] | None = None,
) -> nn.Module:
    """Freeze a model and attach PEFT LoRA adapters."""

    from peft import LoraConfig, get_peft_model

    adapted = get_peft_model(
        model,
        LoraConfig(**lora_config_dict(config, model_name, target_modules)),
    )
    adapted.print_trainable_parameters()
    return adapted


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge trained adapters so existing backbone checkpoint loaders still work."""

    return model.merge_and_unload()
