from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ccvnet.config import load_config
from ccvnet.ablation import descriptor_attribution
from ccvnet.experiments import main_baseline, moe_early_cycle, small_cycle, small_data, transfer


EXPERIMENTS = {
    "main_baseline": main_baseline.run,
    "major-main_baseline": main_baseline.run_major,
    "ablation-main_baseline": main_baseline.run_ablation,
    "all-main_baseline": main_baseline.run_all,
    "small_data": small_data.run,
    "major-small_data": small_data.run_major,
    "ablation-small_data": small_data.run_ablation,
    "all-small_data": small_data.run_all,
    "small_cycle": small_cycle.run,
    "major-small_cycle": small_cycle.run_major,
    "ablation-small_cycle": small_cycle.run_ablation,
    "all-small_cycle": small_cycle.run_all,
    "moe_baseline": moe_early_cycle.run_moe_baseline,
    "ablation-moe_baseline": moe_early_cycle.run_moe_baseline,
    "moe_small_data": moe_early_cycle.run_moe_small_data,
    "ablation-moe_small_data": moe_early_cycle.run_moe_small_data,
    "moe_early_cycle": moe_early_cycle.run,
    "ablation-moe_early_cycle": moe_early_cycle.run_ablation,
    "all-moe_early_cycle": moe_early_cycle.run_all,
    "transfer": transfer.run,
    "major-transfer": transfer.run_major,
    "ablation-transfer": transfer.run_ablation,
    "all-transfer": transfer.run_all,
    "descriptor_attribution": descriptor_attribution.run,
}

DEFAULT_CONFIGS = {
    "main_baseline": "configs/major/main_baseline.yaml",
    "major-main_baseline": "configs/major/main_baseline.yaml",
    "ablation-main_baseline": "configs/ablation/main_baseline.yaml",
    "all-main_baseline": "configs/all/main_baseline.yaml",
    "small_data": "configs/major/small_data.yaml",
    "major-small_data": "configs/major/small_data.yaml",
    "ablation-small_data": "configs/ablation/small_data.yaml",
    "all-small_data": "configs/all/small_data.yaml",
    "small_cycle": "configs/major/small_cycle.yaml",
    "major-small_cycle": "configs/major/small_cycle.yaml",
    "ablation-small_cycle": "configs/ablation/small_cycle.yaml",
    "all-small_cycle": "configs/all/small_cycle.yaml",
    "moe_baseline": "configs/ablation/moe_baseline.yaml",
    "ablation-moe_baseline": "configs/ablation/moe_baseline.yaml",
    "moe_small_data": "configs/ablation/moe_small_data.yaml",
    "ablation-moe_small_data": "configs/ablation/moe_small_data.yaml",
    "moe_early_cycle": "configs/ablation/moe_early_cycle.yaml",
    "ablation-moe_early_cycle": "configs/ablation/moe_early_cycle.yaml",
    "all-moe_early_cycle": "configs/ablation/moe_early_cycle.yaml",
    "transfer": "configs/major/transfer.yaml",
    "major-transfer": "configs/major/transfer.yaml",
    "ablation-transfer": "configs/ablation/transfer.yaml",
    "all-transfer": "configs/all/transfer.yaml",
    "descriptor_attribution": "configs/ablation/small_cycle.yaml",
}


def default_config_for_experiment(experiment: str) -> Path:
    return REPO_ROOT / DEFAULT_CONFIGS.get(experiment, "configs/default.yaml")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def normalize_repo_paths(config: dict) -> dict:
    paths_cfg = config.setdefault("paths", {})
    for key in ["data_dir", "processed_dir", "results_dir"]:
        if key in paths_cfg:
            paths_cfg[key] = str(resolve_path(paths_cfg[key]))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CCVNet training experiments.")
    parser.add_argument("--config", default=None, help="Path to YAML config. Defaults to the experiment-specific config.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="main_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config) if args.config else default_config_for_experiment(args.experiment).resolve()
    config = normalize_repo_paths(load_config(config_path))
    config.setdefault("runtime", {})["config_path"] = str(config_path)
    config["runtime"]["repo_root"] = str(REPO_ROOT)
    EXPERIMENTS[args.experiment](config)


if __name__ == "__main__":
    main()
