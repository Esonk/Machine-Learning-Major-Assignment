import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from breast_cancer_core import FEATURE_COLUMNS, get_project_root, predict_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict benign or malignant breast cancer sample with the best PyTorch model.")
    parser.add_argument(
        "--features",
        nargs=30,
        type=float,
        help="Thirty numeric feature values in the same order as the dataset columns.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Use one row from the local CSV when --features is not provided. Default: 0.",
    )
    return parser.parse_args()


def load_features_from_csv(row_index: int) -> tuple[list[float], str]:
    data_path = get_project_root() / "数据集" / "breast cancer.csv"
    df = pd.read_csv(data_path)
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"row-index must be in [0, {len(df) - 1}]")
    row = df.iloc[row_index]
    features = [float(row[col]) for col in FEATURE_COLUMNS]
    true_label = "恶性" if row["diagnosis"] == "M" else "良性"
    return features, true_label


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    project_root = get_project_root()
    model_path = project_root / "results" / "best_model.pt"
    stats_path = project_root / "results" / "preprocess_stats.json"
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError("Please run train_models.py first to create best_model.pt and preprocess_stats.json.")

    if args.features is None:
        features, true_label = load_features_from_csv(args.row_index)
        source = f"dataset row {args.row_index}, true label: {true_label}"
    else:
        features = args.features
        source = "command-line input"

    result = predict_features(features, model_path=model_path, stats_path=stats_path)
    print(json.dumps({"source": source, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
