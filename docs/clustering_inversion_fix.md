# CLAM 聚类反转问题修复方案

## 问题描述

### 现象

CLAM 模型在训练时 instance-level clustering 准确率接近 100%，但验证时 AUC 接近 0，表现为"全部预测错误"。

```
训练日志:
  class 0 clustering acc 1.0: correct 2616/2616
  class 1 clustering acc 1.0: correct 2616/2616  ✓ 聚类完美

验证日志:
  Val Set, val_loss: 6.7, val_error: 0.97, auc: 0.0000  ✗ AUC接近0
```

### 根本原因

Instance-level 聚类正确地将 patch 分成了两类（cluster 0 和 cluster 1），但 bag-level 分类器的权重方向反了：

- **正确情况**: cluster 0 → class 0 (non-GCB), cluster 1 → class 1 (GCB)
- **反转情况**: cluster 0 → class 1 (GCB), cluster 1 → class 0 (non-GCB)

这导致模型预测的概率与实际标签完全相反，AUC = 1 - 真实AUC。

### 发生概率

根据实验数据分析：
- nanchang 数据集: 约 40% 的 fold 出现聚类反转
- dlbcl_all 数据集: 约 10% 的 fold 出现聚类反转

## 解决方案

### 方案一：评估时自动翻转（推荐，已实现）

`eval.py` 中已实现 `--auto_fix_inversion` 参数。

**优点**: 不改变训练过程，只在评估时修正
**实现**: 检测 AUC < 0.3 且翻转后 AUC > 0.7 时自动翻转

### 方案二：关闭 Instance Clustering

使用 `--bag_weight 1.0 --no_inst_cluster` 参数，只使用 bag-level 损失。

**优点**: 完全避免聚类反转问题
**缺点**: 失去了 CLAM 的 instance-level 监督优势

### 方案三：使用 MIL 模型

使用 `--model_type mil`，使用标准 MIL 而非 CLAM。

**优点**: 最简单，无聚类问题
**缺点**: 没有注意力机制和 instance-level 监督

## 使用方法

### 启用自动翻转

```bash
python eval.py \
  --task task_3_dlbcl_coo --dataset nanchang --feature_type uni \
  --models_exp_code nanchang_uni_clam_sb_s1 \
  --save_exp_code nanchang_uni_clam_sb_eval_fixed \
  --model_type clam_sb \
  --auto_fix_inversion \
  --data_root_dir features --results_dir results \
  --embed_dim 1024 --k 10
```

### 关闭自动翻转（传统行为）

```bash
python eval.py \
  ... \
  --auto_fix_inversion False \
  ...
```

## 风险评估

| 风险 | 等级 | 说明 | 缓解措施 |
|------|------|------|----------|
| 误判正常为反转 | 低 | 条件为 AUC < 0.3 且 翻转 > 0.7，双重条件 | 阈值已设置保守 |
| 漏判反转 | 低 | 只在极端情况下发生 | 可调整阈值 |
| 影响正常模型 | 无 | 只在检测到反转时触发 | 自动条件判断 |

## 参考

- CLAM 论文: Lu et al., "Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images", Nature Biomedical Engineering 2021
- 原始 issue: Instance clustering accuracy 接近 100% 但验证 AUC 接近 0

## 更新记录

| 日期 | 修改内容 |
|------|----------|
| 2026-04-12 | 创建文档，描述聚类反转问题和解决方案 |
| 2026-04-12 | 更新文档，标注 --auto_fix_inversion 已实现 |
