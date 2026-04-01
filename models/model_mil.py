# =============================================================================
# 文件功能：基础 MIL（Multiple Instance Learning）模型定义
# 包含两个模型类：
#   - MIL_fc    : 标准二分类 MIL 模型，使用 top-1 实例选择
#   - MIL_fc_mc : 多分类 MIL 扩展，支持 n_classes > 2
# 均使用 Max-Pooling 风格的实例选择策略（选最具代表性的 patch）
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL_fc(nn.Module):
    """
    标准二分类 MIL 全连接模型。
    结构：FC + ReLU + Dropout -> 分类器
    推理策略：计算每个 patch 的软概率，选取正类概率最高的 top-1 patch 作为 bag 代表，
              用该 patch 的 logit 和 softmax 输出作为 bag 级别的预测结果。
    """
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024):
        super().__init__()
        assert n_classes == 2   # 该模型仅支持二分类
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(size[1], n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        # 计算每个 patch 的 softmax 概率，取正类（索引 1）概率最高的 top-k patch
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            # 返回 top-k 实例的特征，用于下游可视化或分析
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    """
    多分类 MIL 全连接模型（n_classes > 2）。
    结构与 MIL_fc 相同，但分类头输出 n_classes 个 logit。
    推理策略：将 K×n_classes 的概率矩阵展平，
              找全局最大值对应的 (patch_idx, class_idx) 作为预测。
    """
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes > 2   # 该模型仅用于多分类场景
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1   # 多分类版本仅支持 top-1
    
    def forward(self, h, return_features=False):       
        h = self.fc(h)
        logits = self.classifiers(h)

        # 展平概率矩阵，找全局最大概率位置
        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        # 将展平索引还原为 (patch_idx, class_idx)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]        # 预测类别索引
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
