# =============================================================================
# 文件功能：特征级数据增强函数
# 提供以下增强方法：
#   - add_gaussian_noise      : 高斯噪声
#   - feature_dropout          : 特征维度 Dropout
#   - patch_dropout            : Patch Dropout
#   - random_subsample_patches : Patch 数量限制
#   - apply_feature_augmentation : 组合入口
# =============================================================================

import torch
import numpy as np


def add_gaussian_noise(features, std=0.02):
    """
    对 patch 特征加高斯噪声。

    Args:
        features: torch.Tensor, shape [N, D]
        std: 高斯噪声标准差

    Returns:
        torch.Tensor: 加噪后的特征
    """
    if std <= 0:
        return features
    noise = torch.randn_like(features) * std
    return features + noise


def feature_dropout(features, drop_prob=0.1):
    """
    随机将特征维度的部分值置零（类似于 dropout）。

    Args:
        features: torch.Tensor, shape [N, D]
        drop_prob: 每个维度被置零的概率

    Returns:
        torch.Tensor: Dropout 后的特征
    """
    if drop_prob <= 0:
        return features
    mask = torch.rand_like(features) > drop_prob
    return features * mask.float()


def patch_dropout(features, keep_ratio=0.8):
    """
    随机保留 bag 中的一部分 patch。

    Args:
        features: torch.Tensor, shape [N, D]
        keep_ratio: 保留 patch 的比例

    Returns:
        torch.Tensor: Dropout 后的特征
    """
    if keep_ratio >= 1.0:
        return features
    n_patches = features.shape[0]
    n_keep = max(1, int(n_patches * keep_ratio))
    indices = torch.randperm(n_patches)[:n_keep]
    return features[indices]


def random_subsample_patches(features, max_patches=512):
    """
    限制 bag 中最多保留 max_patches 个 patch（随机采样）。

    Args:
        features: torch.Tensor, shape [N, D]
        max_patches: 每个 bag 最多保留的 patch 数

    Returns:
        torch.Tensor: 采样后的特征
    """
    if features.shape[0] <= max_patches:
        return features
    indices = torch.randperm(features.shape[0])[:max_patches]
    return features[indices]


def apply_feature_augmentation(features, config, training=True):
    """
    根据配置对特征应用增强。

    增强只在 training=True 时应用，验证/测试返回原始特征。

    Args:
        features: torch.Tensor, shape [N, D]
        config: dict, 包含以下键值：
            - feature_noise_std: 高斯噪声标准差（默认 0.0）
            - feature_dropout: 特征维度 dropout 概率（默认 0.0）
            - patch_keep_ratio: patch 保留比例（默认 1.0）
            - max_patches_per_bag: 最大 patch 数量（默认 None）
        training: bool, 是否训练模式

    Returns:
        torch.Tensor: 增强后的特征
    """
    if not training:
        return features

    if config is None:
        return features

    # 1. Patch 数量限制（先做这个，因为后面的增强是在剩余 patch 上）
    if config.get('max_patches_per_bag') is not None:
        features = random_subsample_patches(features, config['max_patches_per_bag'])

    # 2. Patch Dropout
    if config.get('patch_keep_ratio') is not None and config['patch_keep_ratio'] < 1.0:
        features = patch_dropout(features, config['patch_keep_ratio'])

    # 3. 特征维度 Dropout
    if config.get('feature_dropout') is not None and config['feature_dropout'] > 0:
        features = feature_dropout(features, config['feature_dropout'])

    # 4. 高斯噪声
    if config.get('feature_noise_std') is not None and config['feature_noise_std'] > 0:
        features = add_gaussian_noise(features, config['feature_noise_std'])

    return features