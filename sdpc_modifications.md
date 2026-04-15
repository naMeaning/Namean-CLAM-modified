# SDPC 支持与 DLBCL 任务定制说明

> 最后更新：2026-04-15
> 用途：说明当前仓库中相对于原始 CLAM 的两类关键定制：
> 1. `.sdpc` 全切片支持
> 2. DLBCL COO 任务与多数据集训练流程

## 1. 文档定位

原始 CLAM 主要面向通用 WSI 分类任务。当前仓库在此基础上做了与毕业论文直接相关的定制，主要包括：

- 支持国产扫描仪生成的 `.sdpc` 文件
- 新增 `task_3_dlbcl_coo`
- 支持 `nanchang` / `morph` / `tcga` / `all` 多数据集
- 使用 patient-level split
- `all` 数据集支持 source-aware split
- DLBCL 训练流程引入更保守的默认设置和稳定性策略

## 2. `.sdpc` 支持

### 2.1 背景

原始 CLAM 默认依赖 `openslide` 读取 `.svs`、`.tiff`、`.ndpi` 等常见 WSI 格式。当前项目为了兼容 `.sdpc`，引入了 `opensdpc`。

### 2.2 相关文件

| 文件 | 作用 |
| --- | --- |
| [wsi_core/WholeSlideImage.py](wsi_core/WholeSlideImage.py) | WSI 加载与 patch 读取 |
| [extract_features_fp.py](extract_features_fp.py) | 特征提取时读取 slide |

### 2.3 当前实现要点

当前逻辑是：

- 如果 slide 后缀为 `.sdpc`，使用 `opensdpc.OpenSdpc(...)`
- 否则仍使用 `openslide.open_slide(...)`
- `read_region(...)` 后统一 `.convert('RGB')`，保证后续处理接口一致

### 2.4 依赖

```bash
pip install opensdpc
```

## 3. DLBCL 任务定制

### 3.1 新增任务

当前仓库新增：

- `task_3_dlbcl_coo`

对应分类任务：

- `GCB` vs `non-GCB`

核心入口：

- [main.py](main.py)
- [eval.py](eval.py)
- [create_splits_seq.py](create_splits_seq.py)

### 3.2 当前不是单一数据集配置

早期版本的 DLBCL 代码和文档曾使用单一 CSV 与单一特征目录，但**当前实现已经不是这样**。

当前 DLBCL 训练入口支持：

- `--dataset nanchang`
- `--dataset morph`
- `--dataset tcga`
- `--dataset all`

同时支持：

- `--feature_type resnet`
- `--feature_type uni`

### 3.3 当前数据集映射

| 数据集 | CSV | 特征目录 |
| --- | --- | --- |
| nanchang | [dataset_csv/nanchang_dlbcl.csv](dataset_csv/nanchang_dlbcl.csv) | `features/nanchang_<feature_type>_features` |
| morph | [dataset_csv/dlbcl_morph.csv](dataset_csv/dlbcl_morph.csv) | `features/morph_<feature_type>_features` |
| tcga | [dataset_csv/tcga_dlbcl.csv](dataset_csv/tcga_dlbcl.csv) | `features/tcga_<feature_type>_features` |
| all | [dataset_csv/dlbcl_all.csv](dataset_csv/dlbcl_all.csv) | `features/all_<feature_type>_features` |

## 4. 当前划分策略

### 4.1 patient-level split

当前 DLBCL 任务使用：

- `patient_strat=True`

这意味着：

- 同一患者的不同 slide 不会跨 train/val/test
- 避免了 slide-level 数据泄露

这与部分旧文档中出现的 `patient_strat=False` 已经不同，后者不再是当前标准。

### 4.2 all 数据集的 source-aware split

`all` 数据集包含多来源样本：

- `nanchang`
- `morph`
- `tcga`

当前 [create_splits_seq.py](create_splits_seq.py) 和 [dataset_modules/dataset_generic.py](dataset_modules/dataset_generic.py) 已支持 source-aware split，核心思想是：

- 不只按 label 分层
- 还按 `(label, source)` 组合分层

目的：

- 避免某一 fold 的 source 分布失衡
- 降低多源混合实验的偶然性

## 5. 当前 DLBCL 训练增强点

当前 DLBCL 任务在代码层面已经加入多项增强，不应再按旧版 CLAM 默认值理解：

- `monitor_metric = val_auc`
- 更保守的默认 dropout / reg / lr
- feature-level augmentation
- bag 级 patch 采样与 dropout
- PCA 降维
- warmup bag-only epochs
- attention entropy regularization
- label smoothing

相关代码：

- [main.py](main.py)
- [utils/core_utils.py](utils/core_utils.py)
- [utils/feature_aug.py](utils/feature_aug.py)
- [utils/pca_utils.py](utils/pca_utils.py)

## 6. 当前推荐命令示例

### 6.1 morph + UNI + CLAM_SB

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

### 6.2 PCA 实验评估

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

## 7. 当前应废弃的旧理解

以下说法若出现在旧文档中，均应视为历史状态：

- DLBCL 只对应 `gcb_vs_nongcb.csv`
- DLBCL 只使用 `dlbcl_resnet_features`
- DLBCL 默认按 slide-level split
- DLBCL 训练默认 `drop_out 0.25 + lr 2e-4`

## 8. 当前结论

当前 `sdpc_modifications.md` 的作用不再只是“记录一次性改动”，而是作为：

- `.sdpc` 兼容说明
- DLBCL 定制任务的准现行说明

后续若代码继续变化，应优先保持本文件与 [operation_guide.md](operation_guide.md) 一致。
