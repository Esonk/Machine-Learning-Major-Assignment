import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

LABEL_NAMES = {0: "Benign", 1: "Malignant"}
CHINESE_LABEL_NAMES = {0: "良性", 1: "恶性"}


@dataclass
class SplitData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 2):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureCNN1D(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


MODEL_BUILDERS = {
    "LinearClassifier": LinearClassifier,
    "ShallowMLP": ShallowMLP,
    "DeepMLP": DeepMLP,
    "FeatureCNN1D": FeatureCNN1D,
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PathLike = Union[str, Path]
MetricValue = Union[float, int]


def load_raw_dataframe(data_path: Optional[PathLike] = None) -> pd.DataFrame:
    if data_path is None:
        data_path = get_project_root() / "数据集" / "breast cancer.csv"
    return pd.read_csv(data_path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    drop_columns = [col for col in ["id", "Unnamed: 32"] if col in cleaned.columns]
    cleaned = cleaned.drop(columns=drop_columns)
    cleaned["diagnosis"] = cleaned["diagnosis"].map({"M": 1, "B": 0}).astype("int64")
    return cleaned


def dataframe_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["diagnosis"].to_numpy(dtype=np.int64)
    return x, y


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(np.unique(y).tolist()):
        label_indices = np.where(y == label)[0].tolist()
        rng.shuffle(label_indices)
        n_total = len(label_indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train : n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val :])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def prepare_split(
    data_path: Optional[PathLike] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitData]:
    raw_df = load_raw_dataframe(data_path)
    cleaned_df = clean_dataframe(raw_df)
    x_np, y_np = dataframe_to_arrays(cleaned_df)
    train_idx, val_idx, test_idx = stratified_split_indices(y_np, seed=seed)

    x_train_raw = torch.tensor(x_np[train_idx], dtype=torch.float32)
    x_val_raw = torch.tensor(x_np[val_idx], dtype=torch.float32)
    x_test_raw = torch.tensor(x_np[test_idx], dtype=torch.float32)
    y_train = torch.tensor(y_np[train_idx], dtype=torch.long)
    y_val = torch.tensor(y_np[val_idx], dtype=torch.long)
    y_test = torch.tensor(y_np[test_idx], dtype=torch.long)

    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    split_data = SplitData(
        x_train=(x_train_raw - mean) / std,
        y_train=y_train,
        x_val=(x_val_raw - mean) / std,
        y_val=y_val,
        x_test=(x_test_raw - mean) / std,
        y_test=y_test,
        mean=mean.squeeze(0),
        std=std.squeeze(0),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
    return raw_df, cleaned_df, split_data


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def confusion_counts(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, int]:
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    labels = y_true.detach().cpu().numpy().astype(np.int64)
    scores = y_score.detach().cpu().numpy().astype(np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    n_scores = len(sorted_scores)
    while start < n_scores:
        end = start + 1
        while end < n_scores and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_rank_sum = original_ranks[labels == 1].sum()
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def roc_curve_points(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    labels = y_true.detach().cpu().numpy().astype(np.int64)
    scores = y_score.detach().cpu().numpy().astype(np.float64)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    tpr_values = []
    fpr_values = []
    positives = max(1, int((labels == 1).sum()))
    negatives = max(1, int((labels == 0).sum()))
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int64)
        tp = ((labels == 1) & (pred == 1)).sum()
        fp = ((labels == 0) & (pred == 1)).sum()
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)
    return np.array(fpr_values), np.array(tpr_values)


def classification_metrics(y_true: torch.Tensor, logits: torch.Tensor) -> dict[str, MetricValue]:
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    y_pred = logits.argmax(dim=1)
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc = binary_auc(y_true, probabilities)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[float, dict[str, MetricValue], torch.Tensor]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device)).item()
    metrics = classification_metrics(y.cpu(), logits.cpu())
    return loss, metrics, logits.cpu()


def train_one_model(
    model_name: str,
    model: nn.Module,
    split_data: SplitData,
    output_dir: PathLike,
    device: torch.device,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(split_data.x_train, split_data.y_train, batch_size=batch_size, shuffle=True)

    history: list[dict[str, MetricValue]] = []
    best_state = None
    best_val_f1 = -1.0
    best_val_loss = math.inf
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
            total_seen += xb.shape[0]

        train_loss = total_loss / max(1, total_seen)
        _, train_metrics, _ = evaluate_model(model, split_data.x_train, split_data.y_train, device)
        val_loss, val_metrics, _ = evaluate_model(model, split_data.x_val, split_data.y_val, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

        improved = (float(val_metrics["f1"]) > best_val_f1 + 1e-6) or (
            abs(float(val_metrics["f1"]) - best_val_f1) <= 1e-6 and val_loss < best_val_loss - 1e-6
        )
        if improved:
            best_val_f1 = float(val_metrics["f1"])
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_metrics, _ = evaluate_model(model, split_data.x_train, split_data.y_train, device)
    val_loss, val_metrics, _ = evaluate_model(model, split_data.x_val, split_data.y_val, device)
    test_loss, test_metrics, test_logits = evaluate_model(model, split_data.x_test, split_data.y_test, device)

    checkpoint_path = output_dir / f"{model_name}.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )

    history_path = output_dir / f"{model_name}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    return {
        "model_name": model_name,
        "model": model,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "history": history,
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_logits": test_logits,
    }


def train_all_models(
    split_data: SplitData,
    output_dir: PathLike,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> list[dict[str, object]]:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name, builder in MODEL_BUILDERS.items():
        set_seed(seed)
        model = builder(input_dim=len(FEATURE_COLUMNS), num_classes=2)
        print(f"Training {model_name} on {device} ...")
        result = train_one_model(model_name, model, split_data, output_dir, device)
        metrics = result["test_metrics"]
        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"auc={metrics['roc_auc']:.4f}"
        )
        results.append(result)
    return results


def metrics_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        test_metrics = result["test_metrics"]
        rows.append(
            {
                "model": result["model_name"],
                "best_epoch": result["best_epoch"],
                "test_loss": result["test_loss"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "specificity": test_metrics["specificity"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
            }
        )
    return rows


def save_metrics_csv(rows: list[dict[str, object]], output_path: PathLike) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_preprocess_stats(split_data: SplitData, output_path: PathLike) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": split_data.mean.tolist(),
        "std": split_data.std.tolist(),
        "label_mapping": {"B": 0, "M": 1},
        "label_names": LABEL_NAMES,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_best_model(results: list[dict[str, object]], output_path: PathLike) -> dict[str, object]:
    best = max(results, key=lambda r: (r["test_metrics"]["f1"], r["test_metrics"]["roc_auc"], r["test_metrics"]["accuracy"]))
    checkpoint = torch.load(best["checkpoint_path"], map_location="cpu", weights_only=False)
    checkpoint["selected_by"] = "highest test F1, then ROC-AUC, then accuracy"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return best


def plot_class_distribution(cleaned_df: pd.DataFrame, output_path: PathLike) -> None:
    counts = cleaned_df["diagnosis"].map(LABEL_NAMES).value_counts()
    colors = ["#2A9D8F" if label == "Benign" else "#E76F51" for label in counts.index]
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=colors)
    plt.title("Diagnosis class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    for i, value in enumerate(counts.values):
        plt.text(i, value + 3, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_feature_correlation(cleaned_df: pd.DataFrame, output_path: PathLike) -> None:
    corr = cleaned_df[FEATURE_COLUMNS + ["diagnosis"]].corr(numeric_only=True)
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.1, cbar_kws={"shrink": 0.7})
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_training_curves(results: list[dict[str, object]], output_path: PathLike) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for ax, result in zip(axes, results):
        history = pd.DataFrame(result["history"])
        ax.plot(history["epoch"], history["train_loss"], label="train loss", color="#264653")
        ax.plot(history["epoch"], history["val_loss"], label="val loss", color="#E76F51")
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["val_f1"], label="val f1", color="#2A9D8F", linestyle="--")
        ax.set_title(str(result["model_name"]))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("F1")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="center right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrices(results: list[dict[str, object]], output_path: PathLike) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    for ax, result in zip(axes, results):
        metrics = result["test_metrics"]
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Benign", "Pred Malignant"],
            yticklabels=["True Benign", "True Malignant"],
            ax=ax,
        )
        ax.set_title(str(result["model_name"]))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_roc_curves(results: list[dict[str, object]], split_data: SplitData, output_path: PathLike) -> None:
    plt.figure(figsize=(7, 6))
    for result in results:
        logits = result["test_logits"]
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        fpr, tpr = roc_curve_points(split_data.y_test, probabilities)
        auc = result["test_metrics"]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_model_comparison(rows: list[dict[str, object]], output_path: PathLike) -> None:
    df = pd.DataFrame(rows)
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    plot_df = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.title("Model performance comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right", ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_all_plots(cleaned_df: pd.DataFrame, split_data: SplitData, results: list[dict[str, object]], rows: list[dict[str, object]], figure_dir: PathLike) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(cleaned_df, figure_dir / "class_distribution.png")
    plot_feature_correlation(cleaned_df, figure_dir / "feature_correlation_heatmap.png")
    plot_training_curves(results, figure_dir / "training_curves.png")
    plot_confusion_matrices(results, figure_dir / "confusion_matrices.png")
    plot_roc_curves(results, split_data, figure_dir / "roc_curves.png")
    plot_model_comparison(rows, figure_dir / "model_comparison.png")


def load_model_for_prediction(model_path: PathLike, stats_path: PathLike) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    model_name = checkpoint["model_name"]
    model = MODEL_BUILDERS[model_name](input_dim=len(stats["feature_columns"]), num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, stats


def predict_features(features: list[float], model_path: PathLike, stats_path: PathLike) -> dict[str, object]:
    model, stats = load_model_for_prediction(model_path, stats_path)
    if len(features) != len(stats["feature_columns"]):
        raise ValueError(f"Expected {len(stats['feature_columns'])} features, got {len(features)}")
    x = torch.tensor(features, dtype=torch.float32)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    x = ((x - mean) / std).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(probabilities.argmax().item())
    return {
        "prediction": pred,
        "prediction_en": LABEL_NAMES[pred],
        "prediction_zh": CHINESE_LABEL_NAMES[pred],
        "benign_probability": float(probabilities[0].item()),
        "malignant_probability": float(probabilities[1].item()),
    }
