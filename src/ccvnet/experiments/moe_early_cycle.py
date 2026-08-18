from __future__ import annotations

from ccvnet.experiments.paper_training import run_moe_early_cycle, run_moe_baseline_benchmark, run_moe_small_data_benchmark
from ccvnet.model_registry import model_spec_to_dict, paper_experiment_model_specs


EXPERIMENT_NAME = "moe_early_cycle"


def model_plan(config: dict, result_set: str = "ablation") -> list[dict]:
    specs = paper_experiment_model_specs(config, experiment_name=EXPERIMENT_NAME, result_set=result_set)
    rows = []
    for spec in specs:
        row = model_spec_to_dict(spec)
        row.update({"result_set": result_set, "status": "scheduled"})
        rows.append(row)
    return rows


def run_ablation(config: dict):
    return run_moe_early_cycle(config, result_set="ablation")


def run_all(config: dict):
    return {"ablation": run_ablation(config)}


def run(config: dict):
    return run_ablation(config)



def run_moe_baseline(config: dict):
    return run_moe_baseline_benchmark(config, result_set="ablation")


def run_moe_small_data(config: dict):
    return run_moe_small_data_benchmark(config, result_set="ablation")
