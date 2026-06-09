from pathlib import Path

import torch

from breast_cancer_core import (
    get_project_root,
    metrics_rows,
    prepare_split,
    save_all_plots,
    save_best_model,
    save_metrics_csv,
    save_preprocess_stats,
    set_seed,
    train_all_models,
)


def main() -> None:
    project_root = get_project_root()
    data_path = project_root / "数据集" / "breast cancer.csv"
    results_dir = project_root / "results"
    figure_dir = project_root / "结果图"
    seed = 42

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_df, cleaned_df, split_data = prepare_split(data_path=data_path, seed=seed)

    print("Breast cancer classification with PyTorch")
    print(f"Dataset path: {data_path}")
    print(f"Raw shape: {raw_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Train/Val/Test: {len(split_data.y_train)}/{len(split_data.y_val)}/{len(split_data.y_test)}")
    print(f"Label counts: B={(cleaned_df['diagnosis'] == 0).sum()}, M={(cleaned_df['diagnosis'] == 1).sum()}")
    print(f"Device: {device}")

    results = train_all_models(split_data, output_dir=results_dir, seed=seed, device=device)
    rows = metrics_rows(results)
    save_metrics_csv(rows, results_dir / "model_metrics.csv")
    save_preprocess_stats(split_data, results_dir / "preprocess_stats.json")
    best = save_best_model(results, results_dir / "best_model.pt")
    save_all_plots(cleaned_df, split_data, results, rows, figure_dir)

    print("\nModel comparison on test set:")
    for row in rows:
        print(
            f"{row['model']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"specificity={row['specificity']:.4f}, "
            f"f1={row['f1']:.4f}, "
            f"roc_auc={row['roc_auc']:.4f}"
        )
    print(f"\nBest model: {best['model_name']}")
    print(f"Saved metrics: {results_dir / 'model_metrics.csv'}")
    print(f"Saved best model: {results_dir / 'best_model.pt'}")
    print(f"Saved figures: {figure_dir}")


if __name__ == "__main__":
    main()
