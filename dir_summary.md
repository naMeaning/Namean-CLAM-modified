# 项目路径摘要

> 最后更新：2026-04-15
> 用途：快速查看当前仓库的关键目录和主要产物位置。

## 1. 根目录

```
/home/shanyiye/CLAM/
```

## 2. 关键目录

| 目录 | 路径 | 说明 |
| --- | --- | --- |
| 原始切片 | `/home/shanyiye/CLAM/raw_slides/` | 当前仓库中基本为空，原始 WSI 可能在仓库外部 |
| 数据 CSV | `/home/shanyiye/CLAM/dataset_csv/` | DLBCL 各数据集标签表 |
| 特征目录 | `/home/shanyiye/CLAM/features/` | UNI 等特征 |
| 数据划分 | `/home/shanyiye/CLAM/splits/` | 交叉验证划分 |
| 训练结果 | `/home/shanyiye/CLAM/results/` | checkpoint、summary、fold 结果 |
| 评估结果 | `/home/shanyiye/CLAM/eval_results/` | 独立评估输出 |
| 分析结果 | `/home/shanyiye/CLAM/analysis_results/` | 错例、ROC、混淆矩阵等 |
| 热图结果 | `/home/shanyiye/CLAM/heatmaps/` | 热图配置和可视化结果 |
| 文档目录 | `/home/shanyiye/CLAM/docs/` | 中文分析文档与执行方案 |

## 3. 核心脚本

| 脚本 | 路径 | 作用 |
| --- | --- | --- |
| 训练 | `/home/shanyiye/CLAM/main.py` | K-Fold 训练入口 |
| 评估 | `/home/shanyiye/CLAM/eval.py` | 独立评估入口 |
| 划分生成 | `/home/shanyiye/CLAM/create_splits_seq.py` | 生成 patient-level split |
| 特征提取 | `/home/shanyiye/CLAM/extract_features_fp.py` | 快速提特征 |
| patch 提取 | `/home/shanyiye/CLAM/create_patches_fp.py` | 生成 patch 坐标 |
| 热图生成 | `/home/shanyiye/CLAM/create_heatmaps.py` | 注意力热图生成 |
| 结果分析 | `/home/shanyiye/CLAM/analyze_eval_results.py` | 分析评估结果 |

## 4. 当前 DLBCL 相关数据文件

| 文件 | 路径 |
| --- | --- |
| nanchang | `/home/shanyiye/CLAM/dataset_csv/nanchang_dlbcl.csv` |
| morph | `/home/shanyiye/CLAM/dataset_csv/dlbcl_morph.csv` |
| tcga | `/home/shanyiye/CLAM/dataset_csv/tcga_dlbcl.csv` |
| all | `/home/shanyiye/CLAM/dataset_csv/dlbcl_all.csv` |

## 5. 当前 DLBCL 相关特征目录

| 数据集 | 路径 |
| --- | --- |
| nanchang UNI | `/home/shanyiye/CLAM/features/nanchang_uni_features/` |
| morph UNI | `/home/shanyiye/CLAM/features/morph_uni_features/` |
| tcga UNI | `/home/shanyiye/CLAM/features/tcga_uni_features/` |
| all UNI | `/home/shanyiye/CLAM/features/all_uni_features/` |

## 6. 当前重点结果目录

| 实验 | 路径 | 状态 |
| --- | --- | --- |
| morph 基线 | `/home/shanyiye/CLAM/results/morph_uni_clam_sb_s1/` | 已完成 |
| morph v2 | `/home/shanyiye/CLAM/results/morph_uni_clam_sb_v2_s1/` | 已完成 |
| morph strong v1 | `/home/shanyiye/CLAM/results/morph_uni_clam_sb_strong_v1_s1/` | 已完成 |
| morph v3 | `/home/shanyiye/CLAM/results/morph_uni_clam_sb_v3_s1/` | 当前最佳完整结果 |
| nanchang reg | `/home/shanyiye/CLAM/results/nanchang_uni_clam_sb_reg_s1/` | 已完成 |
| all v2 | `/home/shanyiye/CLAM/results/all_uni_clam_sb_v2_s1/` | 未完整收口 |

## 7. 当前重点评估目录

| 评估 | 路径 |
| --- | --- |
| morph UNI 基线评估 | `/home/shanyiye/CLAM/eval_results/EVAL_morph_uni_clam_sb_eval/` |
| nanchang UNI 基线评估 | `/home/shanyiye/CLAM/eval_results/EVAL_nanchang_uni_clam_sb_eval/` |
| nanchang fixed 评估 | `/home/shanyiye/CLAM/eval_results/EVAL_nanchang_eval_fixed/` |
| all UNI 基线评估 | `/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_all_uni_clam_sb_eval/` |

## 8. 当前重点分析目录

| 路径 | 说明 |
| --- | --- |
| `/home/shanyiye/CLAM/analysis_results/model_comparison.csv` | 旧版 DLBCL baseline 对比 |
| `/home/shanyiye/CLAM/analysis_results/CLAM_SB_noWS/` | 旧版分析 |
| `/home/shanyiye/CLAM/analysis_results/CLAM_SB_WS/` | 旧版分析 |
| `/home/shanyiye/CLAM/analysis_results/MIL/` | 旧版分析 |

## 9. 当前说明

本文件只保留真实存在且当前仍有参考价值的路径，不再列出不存在或已废弃的脚本与目录。
