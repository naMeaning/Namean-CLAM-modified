# 消融实验文档

> 说明：本文件保留为历史消融草案与思路整理。
> 当前正式执行标准以 [消融实验执行方案.md](./消融实验执行方案.md) 为准。

## 1. 已有实验结果

### Morph 数据集 v3 版本结果（当前最佳）
- **实验名称**: `morph_uni_clam_sb_v3`
- **Test AUC**: 0.7806 ± 0.1700 (排除异常fold后 0.86)
- **配置**: PCA(256) + CosineAnnealing + SWA + 特征增强 + warmup(15 epochs)

---

## 2. 消融实验清单

### 优先级 P0（核心对比，必须做）

#### A1. MIL Baseline
**目的**: 证明 CLAM 的 attention + 聚类机制比标准 MIL 有效

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task task_3_dlbcl_coo --dataset morph --feature_type uni \
    --exp_code morph_mil_v3 --model_type mil \
    --data_root_dir features --results_dir results \
    --use_pca --pca_dim 256 \
    --early_stopping --monitor_metric val_auc
```

**预期**: MIL 应比 CLAM 差，可证明 attention 机制的必要性

---

#### A2. CLAM 纯 Bag Loss（关闭聚类）
**目的**: 验证 instance-level clustering 损失的贡献

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task task_3_dlbcl_coo --dataset morph --feature_type uni \
    --exp_code morph_clam_bag_only_v3 --model_type clam_sb \
    --data_root_dir features --results_dir results \
    --use_pca --pca_dim 256 \
    --bag_weight 1.0 --no_inst_cluster \
    --early_stopping --monitor_metric val_auc
```

**预期**: 可能略差于完整 CLAM，可证明聚类损失的辅助作用

---

#### A3. 关闭 PCA
**目的**: 验证 PCA 降维是否必要

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task task_3_dlbcl_coo --dataset morph --feature_type uni \
    --exp_code morph_no_pca_v3 --model_type clam_sb \
    --data_root_dir features --results_dir results \
    --early_stopping --monitor_metric val_auc
```

**注意**: 去掉 `--use_pca --pca_dim 256` 参数

---

### 优先级 P1（建议做）

#### A4. 关闭特征增强
**目的**: 验证 feature_noise_std + feature_dropout 的正则化效果

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task task_3_dlbcl_coo --dataset morph --feature_type uni \
    --exp_code morph_no_aug_v3 --model_type clam_sb \
    --data_root_dir features --results_dir results \
    --use_pca --pca_dim 256 \
    --feature_noise_std 0 --feature_dropout 0 \
    --early_stopping --monitor_metric val_auc
```

---

#### A5. 不同 bag_weight 对比
**目的**: 找 bag_loss 和 inst_loss 的最优配比

| bag_weight | 命令 |
|-------------|------|
| 0.5 | `--bag_weight 0.5` |
| 0.7 (当前) | `--bag_weight 0.7` |
| 1.0 | `--bag_weight 1.0` (同 A2) |

---

### 优先级 P2（可选）

#### A6. 关闭 warmup
```bash
--warmup_bag_only_epochs 0
```

#### A7. 关闭 SWA
```bash
# 不加 --use_swa --swa_start_epoch N
```

#### A8. 不同 model_size
```bash
--model_size big  # 当前是 small
```

---

## 3. 实验记录表

| 实验编号 | 实验名称 | Test AUC | Val AUC | 状态 |
|----------|----------|----------|---------|------|
| v3 (baseline) | morph_uni_clam_sb_v3 | 0.7806 | 0.6409 | ✅ 完成 |
| A1 | morph_mil_v3 | - | - | ⬜ 待做 |
| A2 | morph_clam_bag_only_v3 | - | - | ⬜ 待做 |
| A3 | morph_no_pca_v3 | - | - | ⬜ 待做 |
| A4 | morph_no_aug_v3 | - | - | ⬜ 待做 |
| A5 | morph_bag_weight_0.5_v3 | - | - | ⬜ 待做 |

---

## 4. 训练结果位置

所有实验结果将保存在:
```
results/<exp_code>_s1/summary.csv
```

例如: `results/morph_mil_v3_s1/summary.csv`

---

## 5. 评估命令

实验完成后，使用以下命令评估:

```bash
# 基本评估
python eval.py --task task_3_dlbcl_coo --dataset morph --models_exp_code <exp_code>_s1 --save_exp_code <exp_code>_eval --model_type clam_sb

# 带聚类反转修复的评估（推荐）
python eval.py --task task_3_dlbcl_coo --dataset morph --models_exp_code <exp_code>_s1 --save_exp_code <exp_code>_eval_fixed --model_type clam_sb --auto_fix_inversion --ckpt_type auc
```

---

## 6. 论文图表建议

### 表格格式（各方法对比）

| 方法 | Test AUC (Mean±Std) | Val AUC (Mean±Std) | Fold数 |
|------|---------------------|-------------------|--------|
| CLAM v3 (完整) | 0.78±0.17 | 0.64±0.10 | 10 |
| MIL | - | - | - |
| CLAM (bag only) | - | - | - |
| CLAM (no PCA) | - | - | - |
| CLAM (no aug) | - | - | - |

### 消融分析要点

1. **MIL vs CLAM**: 证明 attention + clustering 的必要性
2. **Bag only vs CLAM**: 证明 instance-level 聚类的贡献
3. **PCA 影响**: 降维对性能和速度的权衡
4. **特征增强**: 正则化对防止过拟合的效果
