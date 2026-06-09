# -*- coding: utf-8 -*-
from pathlib import Path

import nbformat as nbf


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    nb_path = code_dir / "breast_cancer_pytorch.ipynb"

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (geo3d)", "language": "python", "name": "geo3d"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    cells = []

    def add_md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def add_code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(text.strip()))

    add_md(
        """
# 基于 PyTorch 的乳腺癌良恶性分类

本实验使用 Kaggle Breast Cancer Wisconsin 数据集，对乳腺癌样本进行良性（Benign）和恶性（Malignant）二分类。实验严格使用 PyTorch 完成模型训练与预测，不使用 scikit-learn。对比的 4 种分类方法分别是线性分类器、浅层 MLP、深层 MLP 和一维卷积网络。
"""
    )

    add_code(
        """
from pathlib import Path
import pandas as pd
import torch
from IPython.display import Image, display

from breast_cancer_core import (
    MODEL_BUILDERS,
    FEATURE_COLUMNS,
    get_project_root,
    metrics_rows,
    predict_features,
    prepare_split,
    save_all_plots,
    save_best_model,
    save_metrics_csv,
    save_preprocess_stats,
    set_seed,
    train_all_models,
)

pd.set_option('display.max_columns', 40)
project_root = get_project_root()
data_path = project_root / '数据集' / 'breast cancer.csv'
results_dir = project_root / 'results'
figure_dir = project_root / '结果图'
seed = 42
set_seed(seed)
print('Project root:', project_root)
print('Data path:', data_path)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"""
    )

    add_md(
        """
## 1. 数据读取与基本信息

原始数据包含 569 个样本、33 个字段。其中 `diagnosis` 是标签列，`M` 表示恶性，`B` 表示良性。`id` 只是样本编号，`Unnamed: 32` 是全空列，因此训练前删除这两列。
"""
    )

    add_code(
        """
raw_df, cleaned_df, split_data = prepare_split(data_path=data_path, seed=seed)
print('Raw shape:', raw_df.shape)
print('Cleaned shape:', cleaned_df.shape)
print('Feature count:', len(FEATURE_COLUMNS))
print('Missing values after cleaning:', int(cleaned_df.isna().sum().sum()))
print('Label counts:')
print(cleaned_df['diagnosis'].map({0: 'Benign', 1: 'Malignant'}).value_counts())
raw_df.head()
"""
    )

    add_md(
        """
## 2. 数据清洗与特征概览

清洗后保留 30 个数值特征。所有特征来自细胞核半径、纹理、周长、面积、平滑度、凹陷度等统计量。标签被映射为 `B=0`、`M=1`。
"""
    )

    add_code(
        """
summary = cleaned_df[FEATURE_COLUMNS].describe().T[['mean', 'std', 'min', 'max']]
summary.head(10)
"""
    )

    add_code(
        """
print('Train samples:', len(split_data.y_train), 'class counts:', torch.bincount(split_data.y_train).tolist())
print('Validation samples:', len(split_data.y_val), 'class counts:', torch.bincount(split_data.y_val).tolist())
print('Test samples:', len(split_data.y_test), 'class counts:', torch.bincount(split_data.y_test).tolist())
print('Standardization mean shape:', tuple(split_data.mean.shape))
print('Standardization std shape:', tuple(split_data.std.shape))
"""
    )

    add_md(
        """
## 3. 四种 PyTorch 分类模型

实验固定随机种子为 42，使用训练集均值和标准差进行标准化。损失函数为交叉熵，优化器为 Adam。每个模型最多训练 300 轮，并根据验证集 F1-score 进行 early stopping。
"""
    )

    add_code(
        """
for name, builder in MODEL_BUILDERS.items():
    model = builder(input_dim=len(FEATURE_COLUMNS), num_classes=2)
    print('\\n' + name)
    print(model)
"""
    )

    add_md(
        """
## 4. 模型训练与保存

本单元会训练 4 个模型，并保存每个模型权重、训练历史、标准化参数、最佳模型和评估指标。
"""
    )

    add_code(
        """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = train_all_models(split_data, output_dir=results_dir, seed=seed, device=device)
rows = metrics_rows(results)
save_metrics_csv(rows, results_dir / 'model_metrics.csv')
save_preprocess_stats(split_data, results_dir / 'preprocess_stats.json')
best = save_best_model(results, results_dir / 'best_model.pt')
save_all_plots(cleaned_df, split_data, results, rows, figure_dir)
print('Best model:', best['model_name'])
"""
    )

    add_md(
        """
## 5. 测试集指标对比

评价指标包括 Accuracy、Precision、Recall、Specificity、F1-score、ROC-AUC 和混淆矩阵四个计数值。恶性样本作为正类。
"""
    )

    add_code(
        """
metrics_df = pd.DataFrame(rows)
metrics_df
"""
    )

    add_md(
        """
## 6. 可视化结果

以下图表展示类别分布、特征相关性、训练曲线、混淆矩阵、ROC 曲线和模型性能对比。
"""
    )

    add_code(
        """
for filename in [
    'class_distribution.png',
    'feature_correlation_heatmap.png',
    'training_curves.png',
    'confusion_matrices.png',
    'roc_curves.png',
    'model_comparison.png',
]:
    print(filename)
    display(Image(filename=str(figure_dir / filename)))
"""
    )

    add_md(
        """
## 7. 单样本预测示例

加载训练过程中选出的最佳模型，对数据集中第 0 行样本进行预测，输出良性概率和恶性概率。
"""
    )

    add_code(
        """
sample = raw_df.iloc[0]
features = [float(sample[col]) for col in FEATURE_COLUMNS]
prediction = predict_features(
    features,
    model_path=results_dir / 'best_model.pt',
    stats_path=results_dir / 'preprocess_stats.json',
)
print('True label:', '恶性' if sample['diagnosis'] == 'M' else '良性')
prediction
"""
    )

    add_md(
        """
## 8. 实验结论

在本次固定划分的测试集上，线性分类器、浅层 MLP 和深层 MLP 都取得了较高的 F1-score 和 ROC-AUC，说明该数据集的 30 个统计特征对良恶性区分具有较强判别能力。一维卷积网络也能完成分类，但由于数据本质是表格特征，不具有严格的一维空间邻接关系，因此效果略低于 MLP 类模型。综合准确率、召回率、F1-score 与模型复杂度，线性分类器可作为稳定基线，MLP 可作为更具表达能力的神经网络方案。
"""
    )

    nb["cells"] = cells
    nbf.write(nb, str(nb_path))
    print(f"Wrote {nb_path}")


if __name__ == "__main__":
    main()
