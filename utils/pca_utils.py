# =============================================================================
# 文件功能：PCA 降维工具
# 提供以下功能：
#   - fit_pca_from_train_split : 在训练集上拟合 PCA（仅 fit 一次，避免信息泄漏）
#   - apply_pca_transform      : 对单个 slide 的特征应用 PCA 变换
# =============================================================================

import torch
import numpy as np
from sklearn.decomposition import PCA


def fit_pca_from_train_split(train_dataset, n_components=256, whiten=False):
    """
    在训练集上拟合 PCA。

    注意：PCA 只在训练集上 fit，验证/测试集仅 transform，避免信息泄漏。

    Args:
        train_dataset: Generic_Split 或类似数据集对象
        n_components: 保留的主成分数量
        whiten: 是否对主成分进行白化（默认 False）

    Returns:
        PCA 模型对象
    """
    all_features = []
    for idx in range(len(train_dataset)):
        slide_id = train_dataset.slide_data['slide_id'].iloc[idx]
        data_dir = train_dataset.data_dir
        full_path = f"{data_dir}/pt_files/{slide_id}.pt"
        features = torch.load(full_path)
        all_features.append(features.cpu().numpy())

    all_features = np.vstack(all_features)
    print(f"Fitting PCA on {all_features.shape[0]} patches, dim={all_features.shape[1]}")

    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(all_features)
    print(f"PCA fitted: {n_components} components, explained var: {np.sum(pca.explained_variance_ratio_):.4f}")
    return pca


def apply_pca_transform(features, pca_model, n_components=None):
    """
    对单个 slide 的特征应用 PCA 变换。

    Args:
        features: torch.Tensor, shape [N, original_dim]
        pca_model: 已拟合的 PCA 模型
        n_components: 保留的维度（默认 None，使用模型训练时的维度）

    Returns:
        torch.Tensor, shape [N, n_components]
    """
    if n_components is not None and n_components < pca_model.n_components_:
        return torch.from_numpy(pca_model.transform(features.cpu().numpy())[:, :n_components])
    return torch.from_numpy(pca_model.transform(features.cpu().numpy()))
