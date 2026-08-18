from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml



def slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def results_root(config: dict[str, Any] | None = None) -> Path:
    config = config or {}
    return Path(config.get("paths", {}).get("results_dir", "results"))



def experiment_cache_dir(
    experiment: str,
    config: dict[str, Any] | None = None,
    *,
    create: bool = False,
) -> Path:
    path = results_root(config) / slugify(experiment)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def model_cache_dir(
    experiment: str,
    spec: ModelSpec,
    config: dict[str, Any] | None = None,
    *,
    split_protocol: str | None = None,
    create: bool = False,
) -> Path:
    parts = [experiment_cache_dir(experiment, config), spec.slug]
    if split_protocol:
        parts.append(slugify(split_protocol))
    path = Path(*map(str, parts))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_frames(cache_dir: Path, frames: dict[str, Any], *, index: bool = False) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(cache_dir / f"{slugify(name)}.csv", index=index)


def safe_read_csv(path: Path) -> Any:
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_frames(cache_dir: Path, frame_names: list[str] | None = None) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {}
    paths = (
        [cache_dir / f"{slugify(name)}.csv" for name in frame_names]
        if frame_names is not None
        else sorted(cache_dir.glob("*.csv"))
    )
    return {path.stem: safe_read_csv(path) for path in paths if path.exists()}


def cache_has_content(cache_dir: Path, required_frames: list[str] | None = None) -> bool:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False
    if required_frames:
        return all((cache_dir / f"{slugify(name)}.csv").exists() for name in required_frames)
    return any(cache_dir.iterdir())


def save_experiment_outputs(
    cache_dir: Path,
    *,
    frames: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if frames:
        save_frames(cache_dir, frames)
    if manifest is not None:
        write_manifest(cache_dir / "cache_manifest.json", manifest)
    if config is not None:
        write_resolved_config(cache_dir / "config_resolved.yaml", config)




