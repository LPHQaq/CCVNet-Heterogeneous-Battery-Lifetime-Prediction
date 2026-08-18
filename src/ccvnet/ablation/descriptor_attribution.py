from __future__ import annotations

from ccvnet.experiments.paper_training import run_descriptor_attribution


def run(config: dict) -> dict:
    """Run the small-cycle descriptor-density attribution ablation."""
    return run_descriptor_attribution(config, result_set="ablation")
