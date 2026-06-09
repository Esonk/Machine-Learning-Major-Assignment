import torch

from breast_cancer_core import (
    FEATURE_COLUMNS,
    LinearClassifier,
    get_project_root,
    prepare_split,
    save_preprocess_stats,
    set_seed,
    train_one_model,
)


def main() -> None:
    seed = 42
    set_seed(seed)
    project_root = get_project_root()
    data_path = project_root / "数据集" / "breast cancer.csv"
    results_dir = project_root / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, split_data = prepare_split(data_path=data_path, seed=seed)
    save_preprocess_stats(split_data, results_dir / "preprocess_stats.json")

    model = LinearClassifier(input_dim=len(FEATURE_COLUMNS), num_classes=2)
    result = train_one_model("LinearClassifier", model, split_data, results_dir, device)
    metrics = result["test_metrics"]
    print("LinearClassifier test metrics")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"specificity: {metrics['specificity']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")
    print(f"roc_auc: {metrics['roc_auc']:.4f}")
    print(f"checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
