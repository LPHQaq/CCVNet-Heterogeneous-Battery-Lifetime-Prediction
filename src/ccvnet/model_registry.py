from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


INPUT_MODES = {"abs", "norm"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    key: str
    input_mode: str
    family: str
    plot_group: str = "major"
    max_epochs: int | None = 220
    patience: int | None = 30
    cache_name: str | None = None
    n_cycles: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode={self.input_mode!r}; expected one of {sorted(INPUT_MODES)}.")

    @property
    def slug(self) -> str:
        return self.cache_name or f"{self.key}_{self.input_mode}"


MAIN_MODEL_SPECS = [
    ModelSpec(
        name="SpectrumNN",
        key="spectrumNN",
        input_mode="norm",
        family="neural_cvd",
        plot_group="major",
        cache_name="spectrumnn_norm",
        notes="SpectrumNN baseline on normalized CVD input.",
    ),
    ModelSpec(
        name="CNN",
        key="cnn",
        input_mode="norm",
        family="neural_cvd",
        plot_group="major",
        cache_name="cnn_norm",
        notes="Plain CNN baseline on normalized CVD input.",
    ),
    ModelSpec(
        name="CCV-basic",
        key="ccv_basic",
        input_mode="abs",
        family="ccvnet",
        plot_group="major",
        cache_name="ccv_basic_abs",
        notes="Main CCVNet model using absolute CVD and descriptor inputs.",
    ),
]


MAIN_ABLATION_MODEL_SPECS = [
    ModelSpec(
        name="SpectrumNN",
        key="spectrumNN",
        input_mode="abs",
        family="neural_cvd",
        plot_group="ablation",
        cache_name="spectrumnn_abs",
        notes="SpectrumNN baseline on absolute CVD input.",
    ),
    ModelSpec(
        name="CNN",
        key="cnn",
        input_mode="abs",
        family="neural_cvd",
        plot_group="ablation",
        cache_name="cnn_abs",
        notes="CNN baseline on absolute CVD input.",
    ),
    ModelSpec(
        name="CCV-norm",
        key="ccv_norm",
        input_mode="norm",
        family="ccvnet",
        plot_group="ablation",
        cache_name="ccv_norm",
        notes="Normalized-input CCVNet ablation.",
    ),
    ModelSpec(
        name="CCV-abs-cycleaware",
        key="ccv_abs_cycleaware",
        input_mode="abs",
        family="ccvnet",
        plot_group="ablation",
        cache_name="ccv_abs_cycleaware",
        notes="Cycle-aware CCVNet ablation using absolute CVD input.",
    ),
]


SMALL_CYCLE_ABLATION_MODEL_SPECS = [
    ModelSpec(
        name="SpectrumNN",
        key="spectrumNN",
        input_mode="abs",
        family="neural_cvd",
        plot_group="ablation",
        cache_name="spectrumnn_abs",
    ),
    ModelSpec(
        name="CNN",
        key="cnn",
        input_mode="abs",
        family="neural_cvd",
        plot_group="ablation",
        cache_name="cnn_abs",
    ),
    ModelSpec(
        name="CCV-norm",
        key="ccv_norm",
        input_mode="norm",
        family="ccvnet",
        plot_group="ablation",
        cache_name="ccv_norm",
    ),
]


DEFAULT_MODEL_SPECS = MAIN_MODEL_SPECS


def model_spec_from_dict(item: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=str(item["name"]),
        key=str(item["key"]),
        input_mode=str(item.get("input_mode", "abs")),
        family=str(item.get("family", "ccvnet")),
        plot_group=str(item.get("plot_group", "major")),
        max_epochs=item.get("max_epochs", 220),
        patience=item.get("patience", 30),
        cache_name=item.get("cache_name"),
        n_cycles=item.get("n_cycles"),
        notes=str(item.get("notes", "")),
    )


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "key": spec.key,
        "input_mode": spec.input_mode,
        "family": spec.family,
        "plot_group": spec.plot_group,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "cache_name": spec.cache_name,
        "n_cycles": spec.n_cycles,
        "notes": spec.notes,
    }


def load_model_specs(config: dict[str, Any] | None = None, experiment: str = "main_baseline") -> list[ModelSpec]:
    config = config or {}
    configured = config.get("experiment", {}).get("models")
    if configured:
        return [model_spec_from_dict(item) for item in configured]
    return paper_model_specs(experiment_name=experiment, result_set=config.get("experiment", {}).get("result_set", "major"))


def training_config_for_spec(base_training: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    train_cfg = dict(base_training)
    if spec.max_epochs is not None:
        train_cfg["max_epochs"] = int(spec.max_epochs)
    if spec.patience is not None:
        train_cfg["patience"] = int(spec.patience)
    if spec.n_cycles is not None:
        train_cfg["n_cycles"] = int(spec.n_cycles)
    rename = {
        "epochs": "max_epochs",
        "learning_rate": "lr",
    }
    for old, new in rename.items():
        if old in train_cfg:
            train_cfg[new] = train_cfg.get(new, train_cfg[old])
            train_cfg.pop(old, None)
    return train_cfg


def select_cvd_input(
    input_mode: str,
    *,
    X_cvd_abs: np.ndarray,
    X_cvd_norm: np.ndarray,
) -> np.ndarray:
    if input_mode == "abs":
        return X_cvd_abs
    if input_mode == "norm":
        return X_cvd_norm
    raise ValueError(f"Unknown input_mode={input_mode!r}.")


def select_value_input(
    input_mode: str,
    *,
    X_value_abs: np.ndarray,
    X_value_norm: np.ndarray,
) -> np.ndarray:
    if input_mode == "abs":
        return X_value_abs
    if input_mode == "norm":
        return X_value_norm
    raise ValueError(f"Unknown input_mode={input_mode!r}.")


def with_cycle_count(spec: ModelSpec, n_cycles: int | None) -> ModelSpec:
    if spec.n_cycles is not None or n_cycles is None:
        return spec
    if spec.key == "ccv_abs_cycleaware":
        return replace(spec, n_cycles=int(n_cycles))
    return spec


def paper_model_specs(*, experiment_name: str, result_set: str = "major") -> list[ModelSpec]:
    result_set = str(result_set).strip().lower()
    if result_set == "major":
        return list(MAIN_MODEL_SPECS)
    if result_set == "ablation":
        if experiment_name == "small_cycle":
            return list(SMALL_CYCLE_ABLATION_MODEL_SPECS)
        return list(MAIN_ABLATION_MODEL_SPECS)
    if result_set == "all":
        return [*paper_model_specs(experiment_name=experiment_name, result_set="major"), *paper_model_specs(experiment_name=experiment_name, result_set="ablation")]
    raise ValueError(f"Unknown result_set={result_set!r}; expected 'major', 'ablation', or 'all'.")


def paper_experiment_model_specs(
    config: dict[str, Any] | None = None,
    *,
    experiment_name: str | None = None,
    result_set: str | None = None,
) -> list[ModelSpec]:
    config = config or {}
    exp_cfg = config.get("experiment", {})
    configured = exp_cfg.get("models")
    if configured:
        return [model_spec_from_dict(item) for item in configured]
    experiment_name = experiment_name or str(exp_cfg.get("name", "main_baseline"))
    result_set = result_set or str(exp_cfg.get("result_set", "major"))
    return paper_model_specs(experiment_name=experiment_name, result_set=result_set)
