from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
PREPROCESS_ROOT = REPO_ROOT / "data" / "preprocess"
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))
from common import run_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw battery cycles into CCVNet CVD inputs.")
    parser.add_argument("--dataset", required=True, choices=["all", "MATR", "HUST", "MICH_Joule", "MICH_JECS", "CALCE", "RWTH", "SDU", "STAN", "TONGJI", "XJTU"])
    parser.add_argument("--rawdata-dir", type=Path, default=None, help="Override rawdata directory for a single dataset.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory for a single dataset.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-feature-csv", action=argparse.BooleanOptionalAction, default=True, help="Save descriptor summary CSV as a preprocessing artifact. Training descriptors are still rebuilt from CVD pkl files.")
    args = parser.parse_args()
    summary = run_preprocess(args.dataset, args.rawdata_dir, args.output_dir, overwrite=args.overwrite, save_feature_csv=args.save_feature_csv)
    print(summary)


if __name__ == "__main__":
    main()

