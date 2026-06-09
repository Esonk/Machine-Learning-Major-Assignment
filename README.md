# 2.3 乳腺癌良恶性分类（PyTorch）

本项目基于 Kaggle Breast Cancer Wisconsin 数据集，使用 PyTorch 对乳腺癌样本进行良性/恶性二分类。项目严格不使用 scikit-learn，数据划分、标准化、模型训练、指标计算和可视化均由 PyTorch、NumPy、Pandas、Matplotlib、Seaborn 完成。

## 项目结构

```text
2.3乳腺癌分类
├─ 数据集
│  └─ breast cancer.csv
├─ 大作业代码
│  ├─ breast_cancer_core.py
│  ├─ train_models.py
│  ├─ predict_one.py
│  └─ breast_cancer_pytorch.ipynb
├─ html格式
│  └─ breast_cancer_pytorch.html
├─ 结果图
├─ results
├─ README.md
└─ requirements.txt
```

## 模型

项目比较 4 种 PyTorch 神经网络结构：

1. LinearClassifier：单层线性分类器
2. ShallowMLP：一层隐藏层多层感知机
3. DeepMLP：加入 BatchNorm 和 Dropout 的深层多层感知机
4. FeatureCNN1D：将 30 个表格特征作为一维序列输入的一维卷积网络

## 运行方式

建议使用已安装 PyTorch 的 `geo3d` 环境：

```bat
conda activate geo3d
cd /d "D:\桌面\机器学习大作业\一.大作业\2代码\2.3乳腺癌分类\大作业代码"
python train_models.py
```

训练完成后会生成：

- `results\model_metrics.csv`
- `results\best_model.pt`
- `results\preprocess_stats.json`
- `结果图\class_distribution.png`
- `结果图\feature_correlation_heatmap.png`
- `结果图\training_curves.png`
- `结果图\confusion_matrices.png`
- `结果图\roc_curves.png`
- `结果图\model_comparison.png`

单样本预测：

```bat
python predict_one.py --row-index 0
```

也可以直接传入 30 个特征值：

```bat
python predict_one.py --features 17.99 10.38 122.8 1001 0.1184 0.2776 0.3001 0.1471 0.2419 0.07871 1.095 0.9053 8.589 153.4 0.006399 0.04904 0.05373 0.01587 0.03003 0.006193 25.38 17.33 184.6 2019 0.1622 0.6656 0.7119 0.2654 0.4601 0.1189
```

## 数据说明

原始数据 569 条样本、33 列。清洗时删除 `id` 和全空列 `Unnamed: 32`，保留 30 个数值特征，标签 `diagnosis` 映射为：

- `B -> 0`：良性
- `M -> 1`：恶性

数据按标签分层划分为训练集 70%、验证集 15%、测试集 15%，随机种子固定为 42。
