# CLAM 项目当前操作指南

> 最后更新：2026-04-15
> 用途：提供与当前代码实现一致的、面向毕业论文主线的操作说明。
> 说明：本文件只保留当前推荐流程。历史命令和旧版基线不再作为首选执行标准。

## 1. 当前项目定位

当前仓库不是单纯的原始 CLAM 复现，而是一个围绕 DLBCL COO 二分类任务定制和持续优化的项目。

当前最重要的任务是：

- 任务：`task_3_dlbcl_coo`
- 标签：`GCB` vs `non-GCB`
- 数据集：`nanchang` / `morph` / `tcga` / `all`
- 特征：以 `UNI` 为主
- 训练框架：`MIL` / `CLAM_SB` / `CLAM_MB`

当前论文主线建议优先围绕：

- `morph` 数据集
- `UNI + CLAM_SB`
- 优化版本与消融实验

## 2. 相关文档

当前建议优先参考以下文档：

- 项目理解总览：[project_understanding.md](project_understanding.md)
- 论文持续工作台：[ongoing_thesis_workspace.md](ongoing_thesis_workspace.md)
- 消融实验正式方案：[docs/消融实验执行方案.md](docs/消融实验执行方案.md)
- SDPC 和 DLBCL 定制说明：[sdpc_modifications.md](sdpc_modifications.md)
- 中文文档索引：[docs/README_Chinese.md](docs/README_Chinese.md)

## 3. 当前数据与目录

### 3.1 数据集 CSV

| 数据集 | CSV 文件 | 说明 |
| --- | --- | --- |
| nanchang | [dataset_csv/nanchang_dlbcl.csv](dataset_csv/nanchang_dlbcl.csv) | 单中心数据 |
| morph | [dataset_csv/dlbcl_morph.csv](dataset_csv/dlbcl_morph.csv) | 形态/Hans 推导标签 |
| tcga | [dataset_csv/tcga_dlbcl.csv](dataset_csv/tcga_dlbcl.csv) | 外部小样本 |
| all | [dataset_csv/dlbcl_all.csv](dataset_csv/dlbcl_all.csv) | 三源合并，含 `source` 列 |

### 3.2 特征目录

| 数据集 | 特征目录 |
| --- | --- |
| nanchang UNI | [features/nanchang_uni_features](features/nanchang_uni_features) |
| morph UNI | [features/morph_uni_features](features/morph_uni_features) |
| tcga UNI | [features/tcga_uni_features](features/tcga_uni_features) |
| all UNI | [features/all_uni_features](features/all_uni_features) |

说明：

- `all_uni_features` 通过软链接整合多源特征
- 当前训练主要直接读取 `pt_files/*.pt`

### 3.3 交叉验证划分目录

| 数据集 | 划分目录 |
| --- | --- |
| nanchang | [splits/task_3_dlbcl_coo_nanchang_100](splits/task_3_dlbcl_coo_nanchang_100) |
| morph | [splits/task_3_dlbcl_coo_morph_100](splits/task_3_dlbcl_coo_morph_100) |
| all | [splits/task_3_dlbcl_coo_all_100](splits/task_3_dlbcl_coo_all_100) |

## 4. 当前推荐流程

### 4.1 WSI patch 坐标提取

使用：

- [create_patches_fp.py](create_patches_fp.py)

示例：

```bash
python create_patches_fp.py \
  --source raw_slides/nanchang_dlbcl \
  --save_dir results/nanchang_dlbcl \
  --patch_size 256 \
  --seg \
  --patch
```

说明：

- `.sdpc` WSI 已支持
- 当前仓库里的 `raw_slides/` 基本为空，若需重跑此步骤，需要确认原始切片位置

### 4.2 特征提取

使用：

- [extract_features_fp.py](extract_features_fp.py)

#### morph UNI 特征提取

```bash
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin

python extract_features_fp.py \
  --data_h5_dir results/morph_dlbcl \
  --data_slide_dir raw_slides/DLBCL-Morphology \
  --csv_path dataset_csv/dlbcl_morph.csv \
  --feat_dir features/morph_uni_features \
  --model_name uni_v1 \
  --batch_size 256 \
  --slide_ext .svs
```

#### nanchang UNI 特征提取

```bash
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin

python extract_features_fp.py \
  --data_h5_dir results/nanchang_dlbcl \
  --data_slide_dir raw_slides/nanchang_dlbcl \
  --csv_path dataset_csv/nanchang_dlbcl.csv \
  --feat_dir features/nanchang_uni_features \
  --model_name uni_v1 \
  --batch_size 256 \
  --slide_ext .sdpc
```

#### all UNI 特征说明

`all` 当前通常不重新单独提取，而是通过：

- [features/all_uni_features](features/all_uni_features)

整合各数据源特征。

### 4.3 生成划分

使用：

- [create_splits_seq.py](create_splits_seq.py)

#### morph

```bash
python create_splits_seq.py \
  --task task_3_dlbcl_coo \
  --dataset morph \
  --seed 1 \
  --k 10 \
  --label_frac 1.0
```

#### nanchang

```bash
python create_splits_seq.py \
  --task task_3_dlbcl_coo \
  --dataset nanchang \
  --seed 1 \
  --k 5 \
  --label_frac 1.0
```

#### all

```bash
python create_splits_seq.py \
  --task task_3_dlbcl_coo \
  --dataset all \
  --seed 1 \
  --k 10 \
  --label_frac 1.0
```

说明：

- `all` 会自动使用 source-aware split
- 当前代码采用 patient-level split，而不是旧版 slide-level split

### 4.4 模型训练

使用：

- [main.py](main.py)

#### 当前推荐基线：morph + UNI + CLAM_SB

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --task task_3_dlbcl_coo \
  --dataset morph \
  --feature_type uni \
  --data_root_dir features \
  --results_dir results \
  --exp_code morph_baseline_v3 \
  --model_type clam_sb \
  --seed 1 \
  --k 10 \
  --lr 5e-5 \
  --drop_out 0.5 \
  --reg 1e-3 \
  --bag_loss ce \
  --monitor_metric val_auc \
  --use_pca \
  --pca_dim 256 \
  --feature_noise_std 0.02 \
  --feature_dropout 0.1 \
  --patch_keep_ratio 0.8 \
  --max_patches_per_bag 512 \
  --warmup_bag_only_epochs 10 \
  --attention_entropy_weight 0.001 \
  --label_smoothing 0.05 \
  --bag_weight 0.7 \
  --B 8
```

#### MIL 对照

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --task task_3_dlbcl_coo \
  --dataset morph \
  --feature_type uni \
  --data_root_dir features \
  --results_dir results \
  --exp_code morph_mil_v3 \
  --model_type mil \
  --seed 1 \
  --k 10 \
  --lr 5e-5 \
  --drop_out 0.5 \
  --reg 1e-3 \
  --bag_loss ce \
  --monitor_metric val_auc \
  --use_pca \
  --pca_dim 256 \
  --feature_noise_std 0.02 \
  --feature_dropout 0.1 \
  --patch_keep_ratio 0.8 \
  --max_patches_per_bag 512 \
  --label_smoothing 0.05
```

更多消融实验见：

- [docs/消融实验执行方案.md](docs/消融实验执行方案.md)

### 4.5 模型评估

使用：

- [eval.py](eval.py)

#### 带 PCA 的实验评估

```bash
python eval.py \
  --task task_3_dlbcl_coo \
  --dataset morph \
  --feature_type uni \
  --data_root_dir features \
  --results_dir results \
  --models_exp_code morph_baseline_v3_s1 \
  --save_exp_code morph_baseline_v3_eval \
  --model_type clam_sb \
  --use_pca \
  --pca_dim 256 \
  --k 10
```

#### 不带 PCA 的实验评估

```bash
python eval.py \
  --task task_3_dlbcl_coo \
  --dataset morph \
  --feature_type uni \
  --data_root_dir features \
  --results_dir results \
  --models_exp_code morph_no_pca_v3_s1 \
  --save_exp_code morph_no_pca_v3_eval \
  --model_type clam_sb \
  --k 10
```

说明：

- 从当前版本开始，训练时启用 PCA 会保存每个 fold 的 `s_<fold>_pca.pkl`
- 评估时若传 `--use_pca --pca_dim ...`，会自动加载对应 fold 的 PCA 模型
- 旧实验若没有保存 PCA 文件，则不能直接做 PCA 一致性评估

### 4.6 结果分析

当前主要结果目录：

- 训练结果：[results](results)
- 评估结果：[eval_results](eval_results)
- 分析结果：[analysis_results](analysis_results)

当前重点结果线：

- `morph_uni_clam_sb_s1`
- `morph_uni_clam_sb_v2_s1`
- `morph_uni_clam_sb_strong_v1_s1`
- `morph_uni_clam_sb_v3_s1`

### 4.7 热图

使用：

- [create_heatmaps.py](create_heatmaps.py)

现有配置目录：

- [heatmaps/configs](heatmaps/configs)

现有热图结果主要对应旧版 DLBCL 基线：

- [heatmaps/heatmap_production_results](heatmaps/heatmap_production_results)
- [heatmaps/heatmap_raw_results](heatmaps/heatmap_raw_results)

当前仓库中尚未看到针对 `morph_uni_clam_sb_v3` 的现成热图产物。

## 5. 当前代码中的关键默认逻辑

以下行为是当前代码真实存在的，不应再按旧文档理解：

### 5.1 DLBCL 任务默认更保守

在 [main.py](main.py) 中，`task_3_dlbcl_coo` 会自动采用更保守的默认参数，例如：

- 更高 dropout
- 更强 weight decay
- 默认关闭 weighted sampling
- 更低学习率

### 5.2 patient-level split

当前 DLBCL 任务使用 patient-level split，不再使用旧版 slide-level split。

### 5.3 all 数据集 source-aware split

`all` 数据集会基于 `(label, source)` 做更稳妥的划分。

### 5.4 PCA 影响模型输入维度

启用 PCA 时：

- 训练模型输入维度会切换为 `pca_dim`
- 评估也必须显式传 `--use_pca --pca_dim`

## 6. 当前不建议直接沿用的旧信息

以下内容如果出现在旧文档中，默认视为历史信息，而不是当前执行标准：

- `gcb_vs_nongcb.csv`
- `patient_strat=False`
- 仅使用 `dlbcl_resnet_features`
- `drop_out 0.25 + lr 2e-4` 作为 DLBCL 当前推荐默认
- 不带 PCA 却评估 PCA 模型

## 7. 后续文档协作建议

当前文档职责建议如下：

- 操作执行：本文件
- 项目认知：`project_understanding.md`
- 论文推进：`ongoing_thesis_workspace.md`
- 消融实验：`docs/消融实验执行方案.md`
- 深度问题分析：`docs/训练效果问题分析与改进建议.md`

## 8. 当前结论

后续如果你要跑实验或写论文，建议默认把本文件视为“当前可执行标准”，而不要再直接照搬旧版长文档中的 DLBCL 命令。
