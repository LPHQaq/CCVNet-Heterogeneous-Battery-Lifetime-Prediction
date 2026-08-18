from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    folder: str
    feature_csv: str | None
    base_dir: Path = Path("data/processed")
    battery_type: str = "NMC"
    nominal_capacity_Ah: float = np.nan
    voltage_window: str | None = None
    default_temperature_C: float = np.nan
    default_charging_rate_C: float = np.nan
    default_discharging_rate_C: float = np.nan
    cvd_dir_name: str = "CVD_curve"
    notes: str = ""

    @property
    def root(self) -> Path:
        return Path(self.base_dir) / self.folder

    @property
    def feature_path(self) -> Path | None:
        return None if self.feature_csv is None else self.root / self.feature_csv

    @property
    def cvd_dir(self) -> Path:
        return self.root / self.cvd_dir_name

    @property
    def voltage_bounds(self) -> tuple[float, float] | tuple[None, None]:
        if not self.voltage_window:
            return (None, None)
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", self.voltage_window)
        if not match:
            return (None, None)
        return (float(match.group(1)), float(match.group(2)))


SDU_PROTOCOL_SPECS = [
    {"protocol_id": 1, "id_range": (1, 8), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 2, "id_range": (9, 16), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 2.0, "excluded": False},
    {"protocol_id": 3, "id_range": (17, 22), "charge_rate_C": 0.5, "charge_cutoff_V": 4.3, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 4, "id_range": (23, 30), "charge_rate_C": 1.0, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 5, "id_range": (31, 38), "charge_rate_C": 1.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 6, "id_range": (39, 44), "charge_rate_C": 2.0, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 7, "id_range": (45, 48), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 2.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 8, "id_range": (49, 52), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 3.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 9, "id_range": (53, 56), "charge_rate_C": 1.0, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 2.0, "excluded": False},
    {"protocol_id": 10, "id_range": (57, 60), "charge_rate_C": 1.0, "charge_cutoff_V": 4.3, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 11, "id_range": (61, 64), "charge_rate_C": 1.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 1.0, "discharge_cutoff_V": 2.0, "excluded": False},
    {"protocol_id": 12, "id_range": (65, 68), "charge_rate_C": 1.5, "charge_cutoff_V": 4.3, "discharge_rate_C": 1.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 13, "id_range": (69, 72), "charge_rate_C": 2.0, "charge_cutoff_V": 4.2, "discharge_rate_C": 3.0, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 14, "id_range": (73, 75), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 0.2, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 15, "id_range": (76, 82), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": 0.5, "discharge_cutoff_V": 3.0, "excluded": False},
    {"protocol_id": 16, "id_range": (83, 86), "charge_rate_C": 0.5, "charge_cutoff_V": 4.2, "discharge_rate_C": np.nan, "discharge_cutoff_V": 3.0, "excluded": True},
]


def default_dataset_configs(base_dir: str | Path = "data/processed") -> list[DatasetConfig]:
    base_dir = Path(base_dir)
    return [
        DatasetConfig("MICH", "MICH(Joule)(NMC)", None, base_dir=base_dir, nominal_capacity_Ah=2.36, voltage_window="3.0-4.2V"),
        DatasetConfig("RWTH", "RWTH(NMC)", None, base_dir=base_dir, nominal_capacity_Ah=1.1, voltage_window="3.5-3.9V", default_temperature_C=25.0, default_charging_rate_C=2.0, default_discharging_rate_C=2.0),
        DatasetConfig("TONGJI", "TONGJI(NMC)", None, base_dir=base_dir, voltage_window="2.5-4.2V"),
        DatasetConfig("XJTU", "XJTU(NMC)", None, base_dir=base_dir, nominal_capacity_Ah=2.0, voltage_window="2.5-4.2V", default_temperature_C=25.0),
        DatasetConfig("SDU", "SDU(NMC)", None, base_dir=base_dir, nominal_capacity_Ah=2.4, voltage_window="2.0-4.3V", default_temperature_C=25.0),
        DatasetConfig("STAN", "STAN(NMC)", None, base_dir=base_dir, nominal_capacity_Ah=1.1, voltage_window="2.7-4.2V", default_temperature_C=30.0),
        DatasetConfig("HUST", "HUST(LFP)", None, base_dir=base_dir, battery_type="LFP", nominal_capacity_Ah=1.1, voltage_window="2.0-3.6V", default_temperature_C=25.0),
        DatasetConfig("MATR", "MATR(LFP)", None, base_dir=base_dir, battery_type="LFP", nominal_capacity_Ah=1.1, voltage_window="2.0-3.6V", default_temperature_C=30.0),
    ]


def parse_rate_token(token: str | float | int | None) -> float:
    if token is None:
        return np.nan
    text = str(token).strip().replace("C", "")
    if not text:
        return np.nan
    if "." in text:
        return float(text)
    if text.startswith("0") and len(text) > 1:
        return float(text) / (10 ** (len(text) - 1))
    return float(text)


def parse_sdu_cell_index(cell: str) -> int | None:
    stem = Path(str(cell)).stem
    match = re.search(r"(?:^|_)Battery_(\d+)$", stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def resolve_sdu_protocol_spec(cell: str) -> dict | None:
    cell_index = parse_sdu_cell_index(cell)
    if cell_index is None:
        return None
    for spec in SDU_PROTOCOL_SPECS:
        id_start, id_end = spec["id_range"]
        if id_start <= cell_index <= id_end:
            return spec
    return None


def resolve_nominal_capacity_ah(cell: str, cfg: DatasetConfig) -> float:
    if np.isfinite(cfg.nominal_capacity_Ah):
        return float(cfg.nominal_capacity_Ah)

    stem = Path(str(cell)).stem
    upper_stem = stem.upper()
    if cfg.dataset_name == "CALCE":
        if "CS" in upper_stem:
            return 1.1
        if "CX" in upper_stem:
            return 1.35
    if cfg.dataset_name == "TONGJI":
        if stem.startswith("Tongji1_") or stem.startswith("Tongji2_"):
            return 3.5
        if stem.startswith("Tongji3_"):
            return 2.5
    return np.nan


def parse_cell_metadata(cell: str, cfg: DatasetConfig) -> dict:
    text = str(cell)
    stem = Path(text).stem
    v_min, v_max = cfg.voltage_bounds
    meta = {
        "dataset_name": cfg.dataset_name,
        "battery_type": cfg.battery_type,
        "nominal_capacity_Ah": resolve_nominal_capacity_ah(text, cfg),
        "operation_temperature_C": cfg.default_temperature_C,
        "charging_rate_C": cfg.default_charging_rate_C,
        "discharging_rate_C": cfg.default_discharging_rate_C,
        "voltage_window": cfg.voltage_window,
        "voltage_min_V": v_min,
        "voltage_max_V": v_max,
        "voltage_width_V": (v_max - v_min) if v_min is not None and v_max is not None else np.nan,
        "soc_window": None,
        "protocol": None,
        "condition_group": cfg.dataset_name,
    }

    nmc_temp_match = re.search(r"(?:^|_)NMC_(-?\d+(?:\.\d+)?)C(?:_|$)", text)
    if nmc_temp_match:
        meta["operation_temperature_C"] = float(nmc_temp_match.group(1))

    soc_rate_match = re.search(r"_(\d+-\d+)_([\d.]+)-([\d.]+)C(?:_|$)", text)
    if soc_rate_match:
        meta["soc_window"] = soc_rate_match.group(1)
        meta["charging_rate_C"] = float(soc_rate_match.group(2))
        meta["discharging_rate_C"] = float(soc_rate_match.group(3))

    tongji_match = re.search(r"CY(\d+)-([\d.]+)[_\-*]([\d.]+)(?:--|$)", text)
    if tongji_match:
        meta["operation_temperature_C"] = float(tongji_match.group(1))
        meta["charging_rate_C"] = parse_rate_token(tongji_match.group(2))
        meta["discharging_rate_C"] = parse_rate_token(tongji_match.group(3))
        meta["protocol"] = f"CY{tongji_match.group(1)}-{tongji_match.group(2)}_{tongji_match.group(3)}"

    xjtu_match = re.search(r"XJTU_([\d.]+)C", text)
    if xjtu_match:
        meta["charging_rate_C"] = float(xjtu_match.group(1))
        meta["discharging_rate_C"] = float(xjtu_match.group(1))
        meta["protocol"] = f"{xjtu_match.group(1)}C"

    if cfg.dataset_name == "CALCE":
        meta["protocol"] = text.split("_")[1] if "_" in text else text

    if cfg.dataset_name in {"MICH_JECS", "MICH_JOULE", "MICH"}:
        form_match = re.search(r"MICH_([^_]+)", text)
        if form_match:
            meta["protocol"] = form_match.group(1)

    if cfg.dataset_name == "STAN":
        stan_match = re.match(r"Stanford_(Nova_[A-Za-z]+(?:_[A-Za-z]+)?)_(\d+)$", stem)
        if stan_match:
            meta["protocol"] = stan_match.group(1)
            meta["cell_index"] = float(stan_match.group(2))

    if cfg.dataset_name == "SDU":
        sdu_spec = resolve_sdu_protocol_spec(stem)
        if sdu_spec is not None:
            protocol_label = f"SDU_P{sdu_spec['protocol_id']}"
            meta["protocol"] = f"{protocol_label}_excluded" if sdu_spec.get("excluded") else protocol_label
            meta["charging_rate_C"] = float(sdu_spec["charge_rate_C"])
            meta["discharging_rate_C"] = (
                float(sdu_spec["discharge_rate_C"])
                if np.isfinite(sdu_spec["discharge_rate_C"])
                else np.nan
            )
            meta["voltage_window"] = (
                f"{sdu_spec['discharge_cutoff_V']:.1f}-{sdu_spec['charge_cutoff_V']:.1f}V"
            )
            meta["voltage_min_V"] = float(sdu_spec["discharge_cutoff_V"])
            meta["voltage_max_V"] = float(sdu_spec["charge_cutoff_V"])
            meta["voltage_width_V"] = float(
                sdu_spec["charge_cutoff_V"] - sdu_spec["discharge_cutoff_V"]
            )

    temp_label = (
        "T?" if pd.isna(meta["operation_temperature_C"]) else f"T{meta['operation_temperature_C']:g}"
    )
    charge_label = "C?" if pd.isna(meta["charging_rate_C"]) else f"C{meta['charging_rate_C']:g}"
    discharge_label = (
        "D?" if pd.isna(meta["discharging_rate_C"]) else f"D{meta['discharging_rate_C']:g}"
    )
    if cfg.dataset_name == "SDU" and isinstance(meta.get("protocol"), str) and meta["protocol"]:
        meta["condition_group"] = (
            f"{cfg.dataset_name}|{meta['protocol']}|{temp_label}|{charge_label}|"
            f"{discharge_label}|V{meta['voltage_window']}"
        )
    else:
        meta["condition_group"] = f"{cfg.dataset_name}|{temp_label}|{charge_label}|{discharge_label}"
    return meta


def add_metadata_columns(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    metadata = pd.DataFrame([parse_cell_metadata(cell, cfg) for cell in df["cell"]], index=df.index)
    return pd.concat([df, metadata], axis=1)


def load_feature_tables(configs: list[DatasetConfig]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    missing = []
    for cfg in configs:
        if cfg.feature_path is None or not cfg.feature_path.exists():
            missing.append({"dataset_name": cfg.dataset_name, "reason": "feature_csv_missing"})
            continue
        table = pd.read_csv(cfg.feature_path)
        table = add_metadata_columns(table, cfg)
        table["feature_path"] = str(cfg.feature_path)
        tables.append(table)

    feature_df = pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()
    missing_df = pd.DataFrame(missing)
    return feature_df, missing_df


def infer_value_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in feature_df.columns
        if re.match(r"^(vardQ|meandQ|rngdQ|SOH|V_at_min_dQ|centroid_V|width_halfmin|area_)", str(col))
    ]

