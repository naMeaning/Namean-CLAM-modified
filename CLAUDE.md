# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本项目中工作时提供指导。

## 项目概述

CLAM（Clustering-constrained Attention Multiple Instance Learning）是一个用于全切片图像（WSI）分类的弱监督学习框架。它只需要切片级别的标签，并使用注意力机制学习来识别诊断区域。

## 常用命令

### 环境配置
```bash
conda env create -f env.yml
```

### 特征提取
```bash
python extract_features.py --data_dir /path/to/data --csv_path /path/to/csv --feat_dir /path/to/output --model_name resnet50_trunc --batch_size 256
python extract_features_fp.py --data_h5_dir results --data_slide_dir raw_slides --csv_path results/csv_part1.csv --feat_dir features --batch_size 512 --slide_ext .sdpc
```

### 切片块创建
```bash
python create_patches_fp.py --source raw_slides/ --save_dir results/ --patch_size 256 --seg --patch
```

### 训练
```bash
# 基本训练
python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code task_1_tumor_vs_normal_100 --task task_1_tumor_vs_normal --model_type clam_sb

# DLBCL数据集训练 (使用--dataset选择数据集)
python main.py --task task_3_dlbcl_coo --dataset nanchang --exp_code nanchang_clam_sb --model_type clam_sb --k 10 ...
```

主要训练参数：
- `--model_type`: `clam_sb`（单分支）、`clam_mb`（多分支）或 `mil`
- `--task`: 任务名称（如 `task_1_tumor_vs_normal`、`task_2_tumor_subtyping`、`task_3_dlbcl_coo`）
- `--dataset`: DLBCL数据集选择（`nanchang`、`tcga`、`morph`、`all`）
- `--feature_type`: 特征类型选择（`resnet` 或 `uni`）
- `--bag_loss`: 切片级别损失函数（`ce` 或 `svm`）
- `--inst_loss`: 实例级别聚类损失（`svm` 或 `ce`，仅CLAM有效）
- `--bag_weight`: Bag损失权重（1.0=只用bag loss，0.7=标准配置）
- `--no_inst_cluster`: 关闭instance-level聚类
- `--k`: 折数
- `--monitor_metric`: 早停监控指标（`val_auc` 或 `val_loss`，默认 `val_auc`）
- `--save_best_auc_ckpt`: 保存 best_auc 和 best_loss 两套 checkpoint

### 模型选择建议
- `mil`: 最简单的MIL基线，适合调试和对比
- `clam_sb` + `--bag_weight 1.0 --no_inst_cluster`: 纯bag-level分类，排除聚类干扰
- `clam_sb`（标准）: 同时使用bag-level和instance-level损失

### DLBCL 任务推荐配置

DLBCL 数据集（task_3_dlbcl_coo）已内置更保守的默认参数：

| 参数 | DLBCL默认 | 说明 |
|------|-----------|------|
| `drop_out` | 0.5 | 更高 dropout 减少过拟合 |
| `reg` | 1e-3 | 更强权重衰减 |
| `weighted_sample` | False | DLBCL类别接近平衡 |
| `k` (nanchang) | 5 | 50例患者用5折更稳定 |
| `monitor_metric` | val_auc | 按 AUC 选模型 |
| `feature_noise_std` | 0.02 | 特征高斯噪声 |
| `feature_dropout` | 0.1 | 特征维度 Dropout |
| `patch_keep_ratio` | 0.8 | Patch 保留比例 |
| `max_patches_per_bag` | 512 | 每 bag 最大 patch 数 |
| `warmup_bag_only_epochs` | 10 | 前 N 轮只用 bag loss |
| `attention_entropy_weight` | 1e-3 | Attention 熵正则 |
| `label_smoothing` | 0.05 | 标签平滑 |
| `lr` | 5e-5 | 学习率（配合 Cosine Annealing） |
| `use_swa` | False | SWA 权重平均 |

**评估时指定 checkpoint 类型**:
```bash
python eval.py --ckpt_type auc ...  # 使用 best_val_auc checkpoint
python eval.py --ckpt_type loss ... # 使用 best_val_loss checkpoint
```

**DLBCL 标准训练命令（v3 版本，含 Cosine Annealing + SWA + PCA）**:
```bash
# Morph（132 患者，推荐 v3 配置）
python main.py --task task_3_dlbcl_coo --dataset morph --feature_type uni \
    --exp_code morph_uni_clam_sb_v3 --model_type clam_sb \
    --use_pca --pca_dim 256 \
    --warmup_bag_only_epochs 15 --attention_entropy_weight 0.005 \
    --label_smoothing 0.1 --use_swa --swa_start_epoch 15 \
    --early_stopping --monitor_metric val_auc

# ALL（221 患者）
python main.py --task task_3_dlbcl_coo --dataset all --feature_type uni \
    --exp_code all_uni_clam_sb_v3 --model_type clam_sb \
    --use_pca --pca_dim 384 \
    --warmup_bag_only_epochs 10 --attention_entropy_weight 1e-3 \
    --label_smoothing 0.05 --use_swa --swa_start_epoch 10 \
    --model_size big \
    --early_stopping --monitor_metric val_auc

# Nanchang（50 患者）
python main.py --task task_3_dlbcl_coo --dataset nanchang --feature_type uni \
    --exp_code nanchang_uni_clam_sb_v3 --model_type clam_sb \
    --use_pca --pca_dim 256 \
    --warmup_bag_only_epochs 20 --attention_entropy_weight 0.005 \
    --label_smoothing 0.1 --use_swa --swa_start_epoch 20 \
    --early_stopping --monitor_metric val_auc

# 标准训练（自动使用上表所有推荐参数）
python main.py --task task_3_dlbcl_coo --dataset nanchang --feature_type uni \
    --exp_code nanchang_uni_clam_sb --model_type clam_sb \
    --early_stopping --monitor_metric val_auc
```

## 常见问题与解决方案

### 1. CLAM 聚类反转问题

**现象**: 训练时 instance clustering 准确率接近 100%，但验证时 AUC 接近 0

```
训练日志:
  class 0 clustering acc 1.0: correct 2616/2616
  class 1 clustering acc 1.0: correct 2616/2616  ✓ 聚类完美

验证日志:
  Val Set, val_loss: 6.7, val_error: 0.97, auc: 0.0000  ✗ AUC接近0
```

**原因**: Instance-level 聚类正确地将 patch 分成了两类，但 bag-level 分类器的权重方向反了（cluster 0 对应 class 1，cluster 1 对应 class 0）

**解决方案**:

| 方案 | 命令 | 说明 |
|------|------|------|
| MIL 模型 | `--model_type mil` | 无聚类，纯 bag-level 分类 |
| 关闭聚类 | `--bag_weight 1.0 --no_inst_cluster` | 只用 bag loss |
| **自动修正（推荐）** | `--auto_fix_inversion` (eval.py) | 检测到反转时自动翻转预测 |
| 手动翻转 | AUC≈0 时输出 1-AUC | 临时方案 |

**自动修正用法**:
```bash
python eval.py --auto_fix_inversion ... --save_exp_code eval_fixed
```

### 2. 严重过拟合

**现象**: 训练误差接近 0，但验证误差 > 0.3

```
Train Error: 0.0000  ✓ 完美
Val Error: 0.2833~0.7051  ✗ 很差
```

**原因**: 数据量太小（nanchang 只有 50 患者），模型容量过大

**解决方案**:

| 方案 | 参数调整 |
|------|---------|
| 启用特征增强 | `--feature_noise_std 0.02 --feature_dropout 0.1` |
| Patch Dropout | `--patch_keep_ratio 0.8 --max_patches_per_bag 512` |
| 增大 dropout | `--drop_out 0.5` |
| 增大权重衰减 | `--reg 1e-3` |
| Warmup bag-only | `--warmup_bag_only_epochs 10` |
| Attention 熵正则 | `--attention_entropy_weight 1e-3` |
| 标签平滑 | `--label_smoothing 0.05` |
| 减小模型 | `--model_size small` |
| 使用 MIL 基线 | `--model_type mil` |
| 关闭聚类 | `--bag_weight 1.0 --no_inst_cluster` |
| SWA 权重平均 | `--use_swa --swa_start_epoch 10 --swa_lr 1e-5` |
| Cosine LR | 内置（配合 lr=5e-5） |

### 3. 结果不稳定（方差大）

**现象**: 不同 fold 的 AUC 差异巨大（0.1~0.98）

**原因**: 每 fold 测试集太小（~30 切片）

**解决方案**:
- 使用更大数据集
- 多试几个 random seed
- 用 merge 的 all 数据集（221 患者）

### 训练过程分析命令

```bash
# 分析 TensorBoard 日志
python3 << 'EOF'
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def analyze_fold(log_dir, fold=0):
    ea = event_accumulator.EventAccumulator(f"{log_dir}/{fold}/")
    ea.Reload()
    
    val_auc = [e.value for e in ea.Scalars('val/auc')]
    train_error = [e.value for e in ea.Scalars('train/error')]
    val_error = [e.value for e in ea.Scalars('val/error')]
    
    print(f"Fold {fold}:")
    print(f"  Val AUC: max={max(val_auc):.4f}, final={val_auc[-1]:.4f}")
    print(f"  Train Error: {train_error[-1]:.4f}")
    print(f"  Val Error: {val_error[-1]:.4f}")
    
    if max(val_auc) > 0.7 and val_auc[-1] < 0.3:
        print(f"  ⚠️ 过拟合 + 聚类反转")
    elif max(val_auc) < 0.3:
        print(f"  ⚠️ 聚类反转")

analyze_fold('results/dlbcl_all_uni_clam_sb_s1', 0)
EOF
```

### 评估
```bash
python eval.py --drop_out 0.25 --models_exp_code task_1_tumor_vs_normal_100_s1 --save_exp_code task_1_tumor_vs_normal_eval --task task_1_tumor_vs_normal --model_type clam_sb

# DLBCL数据集评估
python eval.py --task task_3_dlbcl_coo --dataset nanchang --models_exp_code nanchang_clam_sb_s1 ...
```

### 数据划分生成
```bash
python create_splits_seq.py --task task_3_dlbcl_coo --seed 1 --label_frac 1 --k 10
```

### 热力图生成
```bash
python create_heatmaps.py --results_dir results --heatmap_dir heatmaps
```

## 架构

项目采用流水线结构：

1. **WSI处理** (`wsi_core/`): WSI读取、组织分割、切片块提取
2. **特征提取** (`extract_features*.py`): 使用预训练模型提取特征（ResNet50、UNI、CONCH）
3. **数据加载** (`dataset_modules/`): MIL数据集类
4. **模型** (`models/`):
   - `model_clam.py`: CLAM_SB（单分支）、CLAM_MB（多分支）
   - `model_mil.py`: 标准MIL（MIL_fc、MIL_fc_mc）
   - `builder.py`: 特征提取器构建器
5. **训练** (`main.py` + `utils/core_utils.py`): K折交叉验证训练
6. **评估** (`eval.py` + `utils/eval_utils.py`): 使用AUC/准确率指标进行模型评估

## 数据格式

- **CSV**: 必须包含 `case_id`、`slide_id`、`label` 列
- **特征**: `.pt` 文件（N个切片块 x 1024维），`.h5` 文件用于坐标
- **数据划分**: 生成在 `splits/<task>_<label_frac>/splits_*.csv`

## 关键路径

- `results/`: 训练输出、切片块、掩码
- `features/`: 提取的特征
- `splits/`: K折数据划分
- `eval_results/`: 评估输出

## 环境变量

对于 UNI/CONCH 病理大模型，需设置：
- `UNI_CKPT_PATH`: UNI检查点路径
- `CONCH_CKPT_PATH`: CONCH检查点路径

### 使用不同病理大模型提取特征

支持多种预训练模型提取WSI特征：

| 模型 | 参数 | 特征维度 | 性能 |
|------|------|----------|------|
| ResNet-50 | `--model_name resnet50_trunc` | 1024 | 基线 |
| UNI | `--model_name uni_v1` | 1024 | SOTA |
| CONCH | `--model_name conch_v1` | 768 | SOTA |

**特征提取示例：**
```bash
# 使用 ResNet-50
python extract_features_fp.py --data_h5_dir results/xxx --data_slide_dir raw_slides/xxx \
  --csv_path dataset_csv/xxx.csv --feat_dir features/xxx_resnet_features \
  --model_name resnet50_trunc --batch_size 512 --slide_ext .svs

# 使用 UNI (需设置环境变量)
export UNI_CKPT_PATH=/home/shanyiye/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin
python extract_features_fp.py ... --model_name uni_v1 ...

# 使用 CONCH (需设置环境变量)
export CONCH_CKPT_PATH=/path/to/conch_checkpoint.pth
python extract_features_fp.py ... --model_name conch_v1 ...
```

**UNI 模型下载与配置：**
```bash
# 下载 UNI 模型
huggingface-cli download MahmoodLab/UNI --local-dir ~/.cache/huggingface/hub/models--MahmoodLab--UNI

# 设置环境变量（写入 ~/.bashrc 以永久生效）
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin
```

**各数据集 UNI 特征提取命令：**
```bash
# nanchang_dlbcl (.sdpc 格式)
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin
python extract_features_fp.py \
    --data_h5_dir results/nanchang_dlbcl/ \
    --data_slide_dir raw_slides/nanchang_dlbcl/ \
    --csv_path dataset_csv/nanchang_dlbcl.csv \
    --feat_dir features/nanchang_uni_features/ \
    --model_name uni_v1 --batch_size 256 --slide_ext .sdpc

# tcga_dlbcl (.svs 格式)
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin
python extract_features_fp.py \
    --data_h5_dir results/tcga_clbcl/ \
    --data_slide_dir raw_slides/TCGA-DLBC/ \
    --csv_path dataset_csv/tcga_dlbcl.csv \
    --feat_dir features/tcga_uni_features/ \
    --model_name uni_v1 --batch_size 256 --slide_ext .svs

# dlbcl_morph (.svs 格式)
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin
python extract_features_fp.py \
    --data_h5_dir results/morph_dlbcl/ \
    --data_slide_dir raw_slides/DLBCL-Morphology/ \
    --csv_path dataset_csv/dlbcl_morph.csv \
    --feat_dir features/morph_uni_features/ \
    --model_name uni_v1 --batch_size 256 --slide_ext .svs
```

**注意事项：**
- 使用 UNI/CONCH 需要申请权限并下载模型
- 特征维度变化后需确保训练参数 `--embed_dim` 与之匹配
- 不同模型提取的特征目录命名建议：`数据集_模型名_features`

## 数据集配置

### 现有数据集

| Task | CSV文件 | 类别 | 切片/患者 | WSI | 特征 | 特征目录 |
|------|---------|------|-----------|-----|------|----------|
| `task_1_tumor_vs_normal` | tumor_vs_normal_dummy_clean.csv | tumor, normal | - | - | - | - |
| `task_2_tumor_subtyping` | tumor_subtyping_dummy_clean.csv | 3亚型 | - | - | - | - |
| `task_3_dlbcl_coo` (nanchang) | nanchang_dlbcl.csv | GCB, non-GCB | 389/50 | ✓ | ✓ ResNet+UNI | features/nanchang_{resnet,uni}_features/ |
| `task_3_dlbcl_coo` (tcga) | tcga_dlbcl.csv | GCB, non-GCB | 39/39 | ✓ | ✓ ResNet+UNI | features/tcga_{resnet,uni}_features/ |
| `task_3_dlbcl_coo` (morph) | dlbcl_morph.csv | GCB, non-GCB | 183/132 | ✓ | ✓ ResNet+UNI | features/morph_{resnet,uni}_features/ |
| `task_3_dlbcl_coo` (all) | dlbcl_all.csv | GCB, non-GCB | 611/221 | ✓ | ✓ | features/all_{resnet,uni}_features/ |

> 注：三个DLBCL数据集共用 `task_3_dlbcl_coo`，通过 `--dataset` 和 `--feature_type` 区分。
>
> `dlbcl_all.csv` 包含 `source` 列（tcga/nanchang/morph），用于 source-aware 数据划分，确保各折 source 分布均衡。

### DLBCL 数据集详情

#### 1. nanchang_dlbcl (南昌DLBCL)
- 来源: 南昌大学附属医院
- WSI格式: `.sdpc`
- 特征状态: ResNet+UNI 均已提取 (389个pt文件)

#### 2. tcga_dlbcl (TCGA DLBCL)
- 来源: TCGA数据库
- WSI格式: `.svs`
- 特征状态: ResNet+UNI 均已提取 (39个pt文件)
- 已清理: 5个无label的WSI

#### 3. dlbcl_morph (DLBCL Morphology)
- 来源: 临床IHC数据，Hans算法推导
- WSI格式: `.svs`
- 特征状态: ResNet+UNI 均已提取 (183个pt文件)
- Hans算法: 基于CD10、BCL6、MUM1推导GCB/non-GCB
- 已清理: 16个无IHC数据的患者

### DLBCL Morphology 数据集说明

数据集位于 `raw_slides/DLBCL-Morphology/`，包含从临床IHC数据通过Hans算法推导的GCB/non-GCB标签：

- **来源**: 原始数据在 `dataset_csv/morph_clinical_data_cleaned.csv`
- **Hans算法**: 基于 CD10、BCL6、MUM1 三个IHC标记物推导分子亚型
- **处理脚本**: `process_dlbcl_clinical.py`

标签推导逻辑（Hans算法）：
- CD10 阳性 → GCB
- CD10 阴性 + BCL6 阴性 → non-GCB
- CD10 阴性 + BCL6 阳性 + MUM1 阴性 → GCB
- CD10 阴性 + BCL6 阳性 + MUM1 阳性 → non-GCB

数据清洗后保留133个患者、185个切片（已删除无IHC数据无法推导的16个患者）。

### 添加新数据集

1. 在 `dataset_csv/` 目录下准备CSV文件（包含 case_id, slide_id, label 列）
2. 在 `main.py` 中添加对应的 task 分支（参考 task_3_dlbcl_coo 的格式）
3. 在 `eval.py` 中添加对应的评估分支（如果需要评估）

### 分开训练多个数据集

三个DLBCL数据集共用 `task_3_dlbcl_coo`，通过 `--dataset` 参数区分：

```bash
# 生成各数据集独立的 splits（会自动创建 dataset-specific 目录）
python create_splits_seq.py --task task_3_dlbcl_coo --dataset nanchang --seed 1 --k 10
python create_splits_seq.py --task task_3_dlbcl_coo --dataset tcga --seed 1 --k 10
python create_splits_seq.py --task task_3_dlbcl_coo --dataset morph --seed 1 --k 10

# 训练时会自动根据 --dataset 查找对应的 splits 目录
# nanchang_dlbcl
python main.py --task task_3_dlbcl_coo --dataset nanchang \
  --exp_code nanchang_clam_sb --data_root_dir features --results_dir results ...

# tcga_dlbcl
python main.py --task task_3_dlbcl_coo --dataset tcga \
  --exp_code tcga_clam_sb --data_root_dir features --results_dir results ...

# dlbcl_morph
python main.py --task task_3_dlbcl_coo --dataset morph \
  --exp_code morph_clam_sb --data_root_dir features --results_dir results ...
```

关键点：
- `--dataset` 参数会同时影响 CSV 选择、特征目录查找和 splits 目录定位
- splits 目录格式：`splits/task_3_dlbcl_coo_<dataset>_100`（如 `splits/task_3_dlbcl_coo_nanchang_100`）
- `main.py` 已自动根据 `--dataset` 推断 split_dir，无需手动指定

## 数据集目录结构

推荐按以下方式组织数据集（每个数据集独立目录）：

```
CLAM/
├── dataset_csv/                    # CSV标签文件
│   ├── my_dataset.csv
│   └── another_dataset.csv
│
├── raw_slides/                     # 原始WSI图像
│   ├── my_dataset/
│   │   ├── slide_001.svs
│   │   └── slide_002.svs
│   └── another_dataset/
│
├── results/                        # patch提取结果
│   ├── my_dataset/
│   │   ├── patches/               # .h5 坐标文件
│   │   ├── masks/                 # .jpg 掩码图
│   │   ├── stitches/              # .jpg 拼接图
│   │   └── process_list_autogen.csv
│   └── another_dataset/
│
├── features/                       # 提取的特征
│   ├── my_dataset/
│   │   ├── slide_001.pt          # 每个slide一个.pt文件
│   │   ├── slide_002.pt
│   │   └── h5_files.csv           # 特征提取自动生成
│   └── another_dataset/
│
├── splits/                        # K折交叉验证划分
│   ├── task_my_dataset_1.0/      # task名称 + label_frac
│   │   ├── splits_0.csv
│   │   ├── splits_1.csv
│   │   └── ...
│   └── task_another_dataset_1.0/
│
└── experiments/                   # 训练输出（可选，自定义）
    └── my_dataset_clam_sb/
```

## 新增数据集完整流程

### 步骤1: 准备CSV文件

在 `dataset_csv/` 目录下创建CSV，必须包含三列：

```csv
case_id,slide_id,label
patient_001,slide_001,0
patient_001,slide_002,1
patient_002,slide_003,0
```

- `case_id`: 患者/样本ID（用于划分数据集时保持同一患者不同时出现在训练/测试集）
- `slide_id`: 切片ID（对应raw_slides/下的文件名，不含扩展名）
- `label`: 类别标签（整数，从0开始）

### 步骤2: 准备WSI图像

将原始WSI文件放入 `raw_slides/数据集名/` 目录，支持格式：`.svs`, `.ndpi`, `.tif`, `.mrxs`, `.sdpc` 等。

### 步骤3: 创建数据划分

```bash
python create_splits_seq.py --task task_my_dataset --seed 1 --k 10
```

生成 `splits/task_my_dataset_1.0/splits_*.csv`

### 步骤4: 提取patch

```bash
python create_patches_fp.py \
  --source raw_slides/my_dataset/ \
  --save_dir results/my_dataset/ \
  --patch_size 256 --seg --patch --stitch
```

### 步骤5: 提取特征

```bash
python extract_features_fp.py \
  --data_h5_dir results/my_dataset/ \
  --data_slide_dir raw_slides/my_dataset/ \
  --csv_path dataset_csv/my_dataset.csv \
  --feat_dir features/my_dataset/ \
  --batch_size 512 --slide_ext .svs
```

### 步骤6: 训练

```bash
python main.py \
  --task task_my_dataset \
  --exp_code my_dataset_clam_sb \
  --results_dir results/my_dataset/ \
  --data_root_dir features/my_dataset/ \
  --model_type clam_sb \
  --k 10 --label_frac 1.0 \
  --drop_out 0.25 --early_stopping --lr 2e-4
```

### 步骤7: 评估

```bash
python eval.py \
  --models_exp_code my_dataset_clam_sb_s1 \
  --save_exp_code my_dataset_clam_sb_eval \
  --task task_my_dataset \
  --model_type clam_sb
```

### 步骤8: 生成热力图

```bash
python create_heatmaps.py \
  --results_dir results/my_dataset/ \
  --heatmap_dir heatmaps/my_dataset/
```

## 参数命名约定

| 参数 | 建议格式 | 示例 |
|------|----------|------|
| task | `task_数据集名` | `task_liver_cancer` |
| exp_code | `数据集_模型_编号` | `liver_clam_sb_001` |
| CSV文件 | `数据集名.csv` | `liver_cancer.csv` |
