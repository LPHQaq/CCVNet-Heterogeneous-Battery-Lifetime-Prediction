from __future__ import annotations

from ccvnet.experiments.paper_training import run_small_cycle
from ccvnet.model_registry import model_spec_to_dict, paper_experiment_model_specs


EXPERIMENT_NAME = "small_cycle"


def model_plan(config: dict, result_set: str = "all") -> list[dict]:
    specs = paper_experiment_model_specs(config, experiment_name=EXPERIMENT_NAME, result_set=result_set)
    rows = []
    for spec in specs:
        row = model_spec_to_dict(spec)
        row.update({"result_set": result_set, "status": "scheduled"})
        rows.append(row)
    return rows


def run_major(config: dict):
    return run_small_cycle(config, result_set="major")


def run_ablation(config: dict):
    return run_small_cycle(config, result_set="ablation")


def run_all(config: dict):
    return {"major": run_major(config), "ablation": run_ablation(config)}


def run(config: dict):
    return run_major(config)
