# -*- coding: utf-8 -*-
from pathlib import Path

import nbformat as nbf


NOTEBOOKS = [
    {
        "model_name": "LinearClassifier",
        "title": "LinearClassifier（线性分类器）",
        "filename": "LinearClassifier（线性分类器）.ipynb",
        "html": "LinearClassifier（线性分类器）.html",
        "description": "线性分类器只包含一个全连接层，是本实验最简单的 PyTorch 基线模型。它用于检验 30 个标准化特征是否已经具备较强线性可分性。",
    },
    {
        "model_name": "ShallowMLP",
        "title": "ShallowMLP（浅层多层感知机）",
        "filename": "ShallowMLP（浅层多层感知机）.ipynb",
        "html": "ShallowMLP（浅层多层感知机）.html",
        "description": "浅层 MLP 使用一层隐藏层和 ReLU 激活函数，在保持结构简单的同时引入非线性表达能力。",
    },
    {
        "model_name": "DeepMLP",
        "title": "DeepMLP（深层多层感知机）",
        "filename": "DeepMLP（深层多层感知机）.ipynb",
        "html": "DeepMLP（深层多层感知机）.html",
        "description": "深层 MLP 使用多层全连接网络，并加入 BatchNorm 和 Dropout，用于观察更强表达能力和正则化对分类结果的影响。",
    },
    {
        "model_name": "FeatureCNN1D",
        "title": "FeatureCNN1D（一维卷积网络）",
        "filename": "FeatureCNN1D（一维卷积网络）.ipynb",
        "html": "FeatureCNN1D（一维卷积网络）.html",
        "description": "一维卷积网络将 30 个表格特征视作一维序列，通过 Conv1d 提取局部特征组合，作为区别于全连接网络的神经网络方法。",
    },
]


def add_common_cells(cells: list, item: dict) -> None:
    model_name = item["model_name"]

    def add_md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def add_code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(text.strip()))

    add_md(
        f"""
# {item['title']}

{item['description']}

本 Notebook 是 2.3 乳腺癌良恶性分类实验的单模型实现。实验使用 PyTorch 完成；数据清洗、分层划分、标准化、指标计算和模型训练逻辑来自同目录下的 `breast_cancer_core.py`。
"""
    )

    add_code(
        """
from pathlib import Path
import pandas as pd
import torch
from IPython.display import Image, display

from breast_cancer_core import (
    FEATURE_COLUMNS,
    MODEL_BUILDERS,
    classification_metrics,
    get_project_root,
    prepare_split,
    predict_features,
    roc_curve_points,
    save_preprocess_stats,
    set_seed,
    train_one_model,
)

pd.set_option('display.max_columns', 40)
seed = 42
set_seed(seed)
project_root = get_project_root()
data_path = project_root / '数据集' / 'breast cancer.csv'
results_dir = project_root / 'results'
figure_dir = project_root / '结果图'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Project root:', project_root)
print('Data path:', data_path)
print('PyTorch version:', torch.__version__)
print('Device:', device)
"""
    )

    add_md(
        """
## 1. 数据读取与预处理

原始数据包含 569 条样本、33 个字段。训练前删除样本编号 `id` 和全空列 `Unnamed: 32`，将标签 `diagnosis` 映射为 `B=0`、`M=1`。数据按标签分层划分为训练集、验证集和测试集，并只使用训练集统计量完成标准化。
"""
    )

    add_code(
        """
raw_df, cleaned_df, split_data = prepare_split(data_path=data_path, seed=seed)
save_preprocess_stats(split_data, results_dir / 'preprocess_stats.json')
print('Raw shape:', raw_df.shape)
print('Cleaned shape:', cleaned_df.shape)
print('Feature count:', len(FEATURE_COLUMNS))
print('Label counts:')
print(cleaned_df['diagnosis'].map({0: 'Benign', 1: 'Malignant'}).value_counts())
print('Train/Val/Test:', len(split_data.y_train), len(split_data.y_val), len(split_data.y_test))
raw_df.head()
"""
    )

    add_md(
        """
## 2. 模型结构

下面构建当前 Notebook 对应的 PyTorch 模型。输入维度为 30，输出维度为 2，分别表示良性和恶性。
"""
    )

    add_code(
        f"""
model_name = '{model_name}'
model = MODEL_BUILDERS[model_name](input_dim=len(FEATURE_COLUMNS), num_classes=2)
print(model)
"""
    )

    add_md(
        """
## 3. 模型训练

训练使用交叉熵损失函数和 Adam 优化器，最大训练 300 轮。验证集 F1-score 不再提升时触发 early stopping，并保存验证集表现最好的模型参数。
"""
    )

    add_code(
        """
result = train_one_model(
    model_name=model_name,
    model=model,
    split_data=split_data,
    output_dir=results_dir,
    device=device,
    batch_size=32,
    max_epochs=300,
    patience=30,
    lr=1e-3,
    weight_decay=1e-4,
)
print('Best epoch:', result['best_epoch'])
print('Checkpoint:', result['checkpoint_path'])
"""
    )

    add_md(
        """
## 4. 训练过程可视化

训练历史记录包含训练损失、训练 F1、验证损失和验证 F1。通过曲线可以观察模型是否收敛，以及是否出现过拟合趋势。
"""
    )

    add_code(
        """
import matplotlib.pyplot as plt

history = pd.DataFrame(result['history'])
fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.plot(history['epoch'], history['train_loss'], label='train loss', color='#264653')
ax1.plot(history['epoch'], history['val_loss'], label='val loss', color='#E76F51')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2 = ax1.twinx()
ax2.plot(history['epoch'], history['val_f1'], label='val f1', color='#2A9D8F', linestyle='--')
ax2.set_ylabel('F1')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.title(f'{model_name} training curve')
plt.tight_layout()
plt.show()
history.tail()
"""
    )

    add_md(
        """
## 5. 测试集评价

测试集指标包括 Accuracy、Precision、Recall、Specificity、F1-score、ROC-AUC 和混淆矩阵。恶性样本作为正类。
"""
    )

    add_code(
        """
metrics = result['test_metrics']
metrics_df = pd.DataFrame([{
    'model': model_name,
    'accuracy': metrics['accuracy'],
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'specificity': metrics['specificity'],
    'f1': metrics['f1'],
    'roc_auc': metrics['roc_auc'],
    'tn': metrics['tn'],
    'fp': metrics['fp'],
    'fn': metrics['fn'],
    'tp': metrics['tp'],
}])
metrics_df
"""
    )

    add_code(
        """
import numpy as np
import seaborn as sns

matrix = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
plt.figure(figsize=(5, 4))
sns.heatmap(
    matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Pred Benign', 'Pred Malignant'],
    yticklabels=['True Benign', 'True Malignant'],
)
plt.title(f'{model_name} confusion matrix')
plt.tight_layout()
plt.show()
"""
    )

    add_code(
        """
logits = result['test_logits']
probabilities = torch.softmax(logits, dim=1)[:, 1]
fpr, tpr = roc_curve_points(split_data.y_test, probabilities)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, label=f'{model_name} AUC={metrics[\"roc_auc\"]:.3f}', color='#2A9D8F')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{model_name} ROC curve')
plt.legend()
plt.tight_layout()
plt.show()
"""
    )

    add_md(
        """
## 6. 单样本预测

下面选取数据集中第 0 行样本进行预测，展示模型输出的良性概率和恶性概率。
"""
    )

    add_code(
        """
sample = raw_df.iloc[0]
features = [float(sample[col]) for col in FEATURE_COLUMNS]
prediction = predict_features(
    features,
    model_path=results_dir / f'{model_name}.pt',
    stats_path=results_dir / 'preprocess_stats.json',
)
print('True label:', '恶性' if sample['diagnosis'] == 'M' else '良性')
prediction
"""
    )

    add_md(
        """
## 7. 小结

本模型完成了从数据读取、预处理、训练、评价到单样本预测的完整流程。最终模型对比见总览 Notebook `breast_cancer_pytorch.ipynb`。
"""
    )


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    for item in NOTEBOOKS:
        nb = nbf.v4.new_notebook()
        nb["metadata"] = {
            "kernelspec": {"display_name": "Python (geo3d)", "language": "python", "name": "geo3d"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
        cells = []
        add_common_cells(cells, item)
        nb["cells"] = cells
        path = code_dir / item["filename"]
        nbf.write(nb, str(path))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
