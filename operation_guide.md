# CLAM 项目完整操作指南

> 本文档提供 CLAM (Clustering-constrained Attention Multiple Instance Learning) 项目的完整操作指南，包括所有步骤的执行命令、参数说明、模型选项等。
> 
> **项目地址**: https://github.com/mahmoodlab/CLAM
> 
> **参考文档**:
> - [directory_docs.md](file:///home/shanyiye/CLAM/directory_docs.md) - 项目目录说明
> - [sdpc_modifications.md](file:///home/shanyiye/CLAM/sdpc_modifications.md) - SDPC 格式支持与 Task 配置修改

---

## 一、环境准备

### 1.1 创建 conda 环境

```bash
# 克隆项目（如果需要）
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM

# 创建 conda 环境
conda env create -f env.yml
conda activate clam
```

### 1.2 额外依赖安装

```bash
# 安装 opensdpc（支持 .sdpc 格式 WSI）
pip install opensdpc

# 可选：安装 timm（使用 UNI/CONCH 时需要）
pip install timm

# 可选：安装病理基础模型（需要申请权限）
# UNI: https://github.com/mahmoodlab/UNI
# CONCH: https://github.com/mahmoodlab/CONCH
```

---

## 二、数据准备

### 2.1 WSI 文件组织

```
raw_slides/                     # 原始 WSI 文件目录
├── slide_001.sdpc             # 支持 .sdpc, .svs, .tiff 等格式
├── slide_002.svs
└── ...
```

### 2.2 CSV 数据集格式

创建 CSV 文件（位于 `dataset_csv/` 目录），格式如下：

```csv
case_id,slide_id,label
patient_001,slide_001,GCB
patient_001,slide_002,non-GCB
patient_002,slide_003,tumor_tissue
...
```

| 列名 | 说明 |
|------|------|
| `case_id` | 患者 ID（用于 patient-level 划分，避免数据泄露）|
| `slide_id` | 切片 ID（文件名，不含扩展名）|
| `label` | 标签（如 `GCB`, `non-GCB`, `tumor_tissue`, `normal_tissue`）|

---

## 三、完整 Pipeline 流程

### 步骤总览

```
WSI 文件 (.svs / .sdpc / .tiff)
    │
    ▼ [步骤1: Patch 切割]
    create_patches_fp.py
    │
    ▼ [步骤2: 特征提取]
    extract_features_fp.py
    │
    ▼ [步骤3: 生成数据划分]
    create_splits_seq.py
    │
    ▼ [步骤4: 模型训练]
    main.py
    │
    ▼ [步骤5: 模型评估]
    eval.py
    │
    ▼ [步骤6: 热图生成]
    create_heatmaps.py
```

---

## 四、步骤详解与命令

### 4.1 步骤1: Patch 切割 (create_patches_fp.py)

从 WSI 图像中切割 patches，提取组织区域坐标。

#### 命令

```bash
python create_patches_fp.py \
    --source raw_slides/ \
    --save_dir results/ \
    --patch_size 256 \
    --seg \
    --patch \
    --stitch
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | str | **必填** | 原始 WSI 文件所在目录 |
| `--save_dir` | str | **必填** | 结果保存目录 |
| `--patch_size` | int | 256 | Patch 的大小（像素）|
| `--step_size` | int | 256 | Patch 采样步长（通常等于 patch_size）|
| `--seg` | flag | False | 是否进行组织分割（segmentTissue）|
| `--patch` | flag | False | 是否切割 patches |
| `--stitch` | flag | False | 是否生成拼接图（stitch image）|
| `--patch_level` | int | 0 | 提取 patch 的金字塔层级（0=最高分辨率）|
| `--preset` | str | None | 预设配置文件（如 `presets/bwh_biopsy.csv`）|
| `--no_auto_skip` | flag | True | 是否跳过已处理的 slides |
| `--process_list` | str | None | 指定处理的 slide 列表 CSV |

#### 输出

```
results/
├── patches/                    # 切割的 patch 图像（可选）
├── masks/                      # 组织掩膜
├── stitches/                   # patch 拼接图
├── process_list_autogen.csv    # 自动生成的 slide 列表
├── csv_part1.csv               # 第一部分 slide 列表
└── csv_part2.csv               # 第二部分 slide 列表
    └── <slide_id>.h5           # 每个 slide 的 patch 坐标
```

---

### 4.2 步骤2: 特征提取 (extract_features_fp.py)

使用预训练模型提取 patch 特征。

#### 命令

```bash
# 使用 ResNet-50（默认）
python extract_features_fp.py \
    --data_h5_dir results \
    --data_slide_dir raw_slides \
    --csv_path results/csv_part1.csv \
    --feat_dir features \
    --batch_size 512 \
    --slide_ext .sdpc

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py   --data_h5_dir results/TCGA-DLBC/   --data_slide_dir raw_slides/TCGA-DLBC/   --csv_path dataset_csv/tcga_dlbcl.csv   --feat_dir features/TCGA-DLBC/   --slide_ext .svs   --batch_size 512

# 使用 UNI 病理大模型
python extract_features_fp.py \
    --data_h5_dir results \
    --data_slide_dir raw_slides \
    --csv_path results/csv_part1.csv \
    --feat_dir features \
    --model_name uni_v1 \
    --batch_size 256 \
    --slide_ext .sdpc

# 使用 CONCH 病理大模型
python extract_features_fp.py \
    --data_h5_dir results \
    --data_slide_dir raw_slides \
    --csv_path results/csv_part1.csv \
    --feat_dir features \
    --model_name conch_v1 \
    --batch_size 256 \
    --slide_ext .sdpc
```

#### UNI/CONCH 模型配置

**下载 UNI 模型：**
```bash
huggingface-cli download MahmoodLab/UNI --local-dir ~/.cache/huggingface/hub/models--MahmoodLab--UNI
```

**设置环境变量：**
```bash
# UNI
export UNI_CKPT_PATH=~/.cache/huggingface/hub/models--MahmoodLab--UNI/pytorch_model.bin

# CONCH
export CONCH_CKPT_PATH=/path/to/conch_checkpoint.pth
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

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_h5_dir` | str | **必填** | patch 坐标 h5 文件所在目录 |
| `--data_slide_dir` | str | **必填** | 原始 WSI 文件目录 |
| `--csv_path` | str | **必填** | slide 列表 CSV 文件路径 |
| `--feat_dir` | str | **必填** | 特征输出目录 |
| `--slide_ext` | str | `.svs` | WSI 文件扩展名（如 `.sdpc`, `.svs`, `.tiff`）|
| `--model_name` | str | `resnet50_trunc` | **特征提取模型**（见下方详表）|
| `--batch_size` | int | 256 | 批处理大小 |
| `--no_auto_skip` | flag | False | 是否跳过已处理的 slides |
| `--target_patch_size` | int | 224 | patch resize 后的尺寸 |

#### 特征提取模型选项

| 模型名称 | 说明 | 特征维度 | 适用场景 |
|----------|------|----------|----------|
| `resnet50_trunc` | ResNet-50 截断版（默认）| 1024 | 通用基线 |
| `uni_v1` | UNI 病理大模型 | 1024 | **推荐**，SOTA 性能 |
| `conch_v1` | CONCH 病理大模型 | 768 | **推荐**，SOTA 性能 |

> **注意**: 使用 `uni_v1` 或 `conch_v1` 需要：
> 1. 申请模型权限（https://github.com/mahmoodlab/UNI）
> 2. 设置环境变量 `UNI_CKPT_PATH` 或 `CONCH_CKPT_PATH`

#### 输出

```
features/
└── <task>_resnet_features/      # 任务对应的特征目录
    ├── pt_files/
    │   ├── <slide_id>.pt       # PyTorch tensor，形状 (N, D)
    └── h5_files/
        └── <slide_id>.h5      # 包含特征和坐标
```

---

### 4.3 步骤3: 生成数据划分 (create_splits_seq.py)

生成 K-Fold 交叉验证的数据划分。

#### 命令

```bash
# 完整标签（100% 数据用于训练）
python create_splits_seq.py \
    --task task_3_dlbcl_coo \
    --seed 1 \
    --label_frac 1.0 \
    --k 10

# 部分标签（用于少样本学习实验）
python create_splits_seq.py \
    --task task_3_dlbcl_coo \
    --seed 1 \
    --label_frac 0.25 \
    --k 10
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | **必填** | 任务名称（见下方详表）|
| `--seed` | int | 1 | 随机种子（保证可复现）|
| `--label_frac` | float | 1.0 | 训练标签比例（1.0=全部，0.75=75%...）|
| `--k` | int | 10 | 交叉验证折数 |
| `--val_frac` | float | 0.1 | 验证集比例 |
| `--test_frac` | float | 0.1 | 测试集比例 |

#### 任务选项 (--task)

| task 名称 | 说明 | 类别数 | CSV 文件 |
|-----------|------|--------|----------|
| `task_1_tumor_vs_normal` | 肿瘤 vs 正常二分类 | 2 | `tumor_vs_normal_dummy_clean.csv` |
| `task_2_tumor_subtyping` | 肿瘤亚型多分类 | 3 | `tumor_subtyping_dummy_clean.csv` |
| `task_3_dlbcl_coo` | DLBCL GCB vs non-GCB 二分类 | 2 | `gcb_vs_nongcb.csv` |

#### 输出

```
splits/
└── <task>_<label_frac*100>/   # 如 task_3_dlbcl_coo_100
    ├── splits_0.csv
    ├── splits_1.csv
    └── ...
    └── splits_9.csv
```

每个 splits_X.csv 文件内容：
```csv
train:patient,val:patient,test:patient
patient_001,patient_010,patient_020
patient_002,patient_011,patient_021
...
```

---

### 4.4 步骤4: 模型训练 (main.py)

训练 CLAM 或 MIL 模型。

#### 命令

##### 4.4.1 CLAM 单分支 (CLAM_SB)

```bash
# CLAM_SB + 带权采样 + 弱监督（默认配置）
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir /home/shanyiye/CLAM/features \
    --results_dir /home/shanyiye/CLAM/results \
    --exp_code dlbcl_gcb_nongcb_clam_sb \
    --model_type clam_sb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --log_data \
    --embed_dim 1024 \
    --weighted_sample

# CLAM_SB + 无弱监督（关闭 instance-level clustering）
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir /home/shanyiye/CLAM/features \
    --results_dir /home/shanyiye/CLAM/results \
    --exp_code dlbcl_gcb_nongcb_clam_sb_nows \
    --model_type clam_sb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --log_data \
    --embed_dim 1024 \
    --no_inst_cluster
```

##### 4.4.2 CLAM 多分支 (CLAM_MB)

```bash
# CLAM_MB（多分支，适用于多分类任务）
# 注：task_3_dlbcl_coo 是二分类任务，通常使用 CLAM_SB
# 如果需要用 CLAM_MB 训练二分类，可参考以下命令：
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir /home/shanyiye/CLAM/features \
    --results_dir /home/shanyiye/CLAM/results \
    --exp_code dlbcl_gcb_nongcb_clam_mb \
    --model_type clam_mb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --log_data \
    --embed_dim 1024
```

##### 4.4.3 CLAM_MB 无 Weighted Sample（对照实验）

```bash
# CLAM_MB 无 weighted sample（对比实验）
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir /home/shanyiye/CLAM/features \
    --results_dir /home/shanyiye/CLAM/results \
    --exp_code dlbcl_gcb_nongcb_clam_mb_nows \
    --model_type clam_mb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --log_data \
    --embed_dim 1024
```

##### 4.4.3 标准 MIL 基线

```bash
# MIL 基线模型
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir /home/shanyiye/CLAM/features \
    --results_dir /home/shanyiye/CLAM/results \
    --exp_code dlbcl_gcb_nongcb_mil \
    --model_type mil \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --log_data \
    --embed_dim 1024
```

#### 参数说明

##### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | **必填** | 任务名称 |
| `--data_root_dir` | str | **必填** | 特征文件根目录 |
| `--results_dir` | str | `./results` | 结果保存目录 |
| `--exp_code` | str | **必填** | 实验代码（用于命名输出目录）|
| `--seed` | int | 1 | 随机种子 |

##### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_type` | str | `clam_sb` | **模型类型**（见下方详表）|
| `--model_size` | str | `small` | 模型规模（`small` 或 `big`）|
| `--drop_out` | float | 0.25 | Dropout 比例 |
| `--embed_dim` | int | 1024 | 特征嵌入维度 |

##### 模型类型选项 (--model_type)

| 模型类型 | 说明 | 适用场景 |
|----------|------|----------|
| `clam_sb` | **CLAM 单分支**（Single Branch）| 二分类任务，推荐 |
| `clam_mb` | **CLAM 多分支**（Multi Branch）| 多分类任务（如肿瘤亚型分型）|
| `mil` | 标准 MIL 基线 | 对比实验、无需注意力可视化 |

##### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr` | float | 1e-4 | 学习率（推荐 1e-4 ~ 5e-4）|
| `--max_epochs` | int | 200 | 最大训练轮数 |
| `--opt` | str | `adam` | 优化器（`adam` 或 `sgd`）|
| `--reg` | float | 1e-5 | 权重衰减（L2 正则化）|

##### K-Fold 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--k` | int | 10 | 交叉验证折数 |
| `--k_start` | int | -1 | 起始折（-1=最后一折）|
| `--k_end` | int | -1 | 结束折（-1=第一折）|
| `--label_frac` | float | 1.0 | 训练标签比例 |

##### 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--bag_loss` | str | `ce` | **Bag 级别损失函数**（见下方详表）|
| `--inst_loss` | str | None | **Instance 级别损失函数**（仅 CLAM）|
| `--bag_weight` | float | 0.7 | Bag 损失权重（1 - bag_weight = instance 损失权重）|
| `--B` | int | 8 | 每个类别采样的正/负样本数量 |

##### 损失函数选项

| 损失函数 | 参数值 | 说明 |
|----------|--------|------|
| 交叉熵 | `ce` | Cross Entropy，最常用 |
| SVM 损失 | `svm` | Hinge Loss，适合分类边界 |

> **推荐配置**:
> - `--bag_loss ce` + `--inst_loss svm`：标准 CLAM 配置
> - `--bag_loss ce` + `--inst_loss ce`：另一种组合
> - `--inst_loss None` 或 `--no_inst_cluster`：关闭 instance-level 聚类损失

##### 其他参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--early_stopping` | flag | 启用早停机制 |
| `--weighted_sample` | flag | 启用带权采样（根据类别数量平衡）|
| `--subtyping` | flag | 启用亚型模式（用于 task_2）|
| `--log_data` | flag | 使用 TensorBoard 记录日志 |
| `--testing` | flag | 调试模式（只运行一个 batch）|

#### 输出

```
results/<exp_code>_s<seed>/
├── logs/                       # TensorBoard 日志
├── checkpoints/               # 模型检查点
│   └── s<fold>/<epoch>.pt
├── summary.csv                # 训练结果汇总
├── splits_<fold>.csv          # 数据划分信息
└── experiment_<exp_code>.txt   # 实验配置记录
```

---

### 4.5 步骤5: 模型评估 (eval.py)

在独立测试集上评估训练好的模型。

#### 命令

```bash
# 评估 CLAM_SB 模型
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --k 10 \
    --models_exp_code dlbcl_gcb_nongcb_clam_sb_s1 \
    --save_exp_code dlbcl_gcb_nongcb_clam_sb_eval \
    --task task_3_dlbcl_coo \
    --model_type clam_sb \
    --results_dir /home/shanyiye/CLAM/results \
    --data_root_dir /home/shanyiye/CLAM/features \
    --embed_dim 1024 \
    --seed 1

# 评估 CLAM_MB 模型
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --k 10 \
    --models_exp_code dlbcl_gcb_nongcb_clam_mb_s1 \
    --save_exp_code dlbcl_gcb_nongcb_clam_mb_eval \
    --task task_3_dlbcl_coo \
    --model_type clam_mb \
    --results_dir /home/shanyiye/CLAM/results \
    --data_root_dir /home/shanyiye/CLAM/features \
    --embed_dim 1024 \
    --seed 1

# 评估 CLAM_MB 无 Weighted Sample 模型
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --k 10 \
    --models_exp_code dlbcl_gcb_nongcb_clam_mb_nows_s1 \
    --save_exp_code dlbcl_gcb_nongcb_clam_mb_nows_eval \
    --task task_3_dlbcl_coo \
    --model_type clam_mb \
    --results_dir /home/shanyiye/CLAM/results \
    --data_root_dir /home/shanyiye/CLAM/features \
    --embed_dim 1024 \
    --seed 1

# 评估特定折
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --fold 5 \
    --models_exp_code dlbcl_gcb_nongcb_clam_sb_s1 \
    --save_exp_code dlbcl_gcb_nongcb_clam_sb_eval_fold5 \
    --task task_3_dlbcl_coo \
    --model_type clam_sb \
    --results_dir /home/shanyiye/CLAM/results \
    --data_root_dir /home/shanyiye/CLAM/features \
    --embed_dim 1024 \
    --seed 1
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--models_exp_code` | str | **必填** | 训练结果目录名（results 下的子目录）|
| `--save_exp_code` | str | **必填** | 评估结果保存名称 |
| `--task` | str | **必填** | 任务名称 |
| `--model_type` | str | `clam_sb` | 模型类型 |
| `--results_dir` | str | `./results` | results 目录路径 |
| `--data_root_dir` | str | **必填** | 特征文件目录 |
| `--embed_dim` | int | 1024 | 特征维度 |
| `--drop_out` | float | 0.25 | Dropout 比例 |
| `--model_size` | str | `small` | 模型规模 |
| `--seed` | int | 1 | 随机种子 |

##### 评估范围控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--k` | int | 10 | 总折数 |
| `--k_start` | int | -1 | 起始折 |
| `--k_end` | int | -1 | 结束折 |
| `--fold` | int | -1 | 单独评估某一折（覆盖 k_start/k_end）|
| `--split` | str | `test` | 评估数据集（`train`/`val`/`test`/`all`）|
| `--micro_average` | flag | False | 多分类时使用 micro-average AUC |

#### 输出

```
eval_results/EVAL_<save_exp_code>/
├── eval_results.csv           # 评估结果
├── eval_experiment_<exp_code>.txt  # 评估配置
└── ...
```

---

### 4.6 步骤6: 热图生成 (create_heatmaps.py)

生成注意力热图，可视化模型关注的区域。

#### 6.1 输入数据格式

热图生成需要以下输入：

1. **训练好的模型 checkpoint**：位于 `results/<exp_code>_s<seed>/s_<fold>_checkpoint.pt`
2. **WSI 原始文件**：位于 `raw_slides/` 目录
3. **Process List CSV**：指定需要生成热图的 slides

**Process List CSV 格式**（`heatmaps/process_lists/` 目录）：

```csv
slide_id,label
slide_001,GCB
slide_002,non-GCB
slide_003,GCB
```

| 列名 | 说明 |
|------|------|
| `slide_id` | 切片 ID（不含扩展名）|
| `label` | 标签（可选，用于分组保存）|

#### 6.2 配置文件详解

创建配置文件（推荐复制模板后修改）：

```bash
cp heatmaps/configs/config_template.yaml heatmaps/configs/dlbcl_heatmap.yaml
```

**配置文件参数说明**：

```yaml
# === 实验参数 ===
exp_arguments:
  n_classes: 2                              # 类别数
  save_exp_code: dlbcl_clam_sb_heatmap    # 输出目录名称
  raw_save_dir: heatmaps/heatmap_raw_results    # 原始结果保存目录
  production_save_dir: heatmaps/heatmap_production_results  # 最终热图保存目录
  batch_size: 256

# === 数据参数 ===
data_arguments:
  data_dir: raw_slides/                    # WSI 文件目录
  process_list: process_list_dlbcl.csv    # 处理列表 CSV
  preset: presets/bwh_biopsy.csv          # 分割预设
  slide_ext: .sdpc                         # WSI 文件扩展名
  label_dict:                               # 标签映射
    GCB: 0
    non-GCB: 1

# === Patch 参数 ===
patching_arguments:
  patch_size: 256
  overlap: 0.5                              # Patch 重叠比例（0.5 = 50% 重叠）
  patch_level: 0
  custom_downsample: 1

# === 编码器参数 ===
encoder_arguments:
  model_name: resnet50_trunc               # 特征提取模型
  target_img_size: 224

# === 模型参数 ===
model_arguments:
  ckpt_path: results/dlbcl_gcb_nongcb_clam_sb_s1/s_0_checkpoint.pt  # 模型 checkpoint 路径
  model_type: clam_sb                      # 模型类型
  initiate_fn: initiate_model
  model_size: small
  drop_out: 0.
  embed_dim: 1024

# === 热图参数 ===
heatmap_arguments:
  vis_level: -1                            # 可视化层级（-1=自动选择）
  alpha: 0.4                               # 热图透明度（0=仅背景，1=仅热图）
  blank_canvas: false                      # 是否使用空白画布
  save_orig: true                          # 是否保存原始图像
  save_ext: jpg                            # 保存格式
  use_ref_scores: true                     # 使用百分位分数
  blur: false                              # 是否高斯模糊
  use_center_shift: true                   # 偏移角点检查
  use_roi: false                           # 是否使用 ROI
  calc_heatmap: true                       # 计算热图
  binarize: false                          # 是否二值化
  binary_thresh: -1                        # 二值化阈值
  custom_downsample: 1
  cmap: jet                                # 颜色映射：jet, viridis, plasma, inferno, magma, coolwarm

# === 采样参数 ===
sample_arguments:
  samples:
    - name: "topk_high_attention"
      sample: true
      seed: 1
      k: 15                                # 保存高注意力 top-k patches
      mode: topk
```

#### 6.3 可视化效果调整选项

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **颜色映射 (cmap)** | | |
| `jet` | 蓝→青→黄→红（经典） | 默认 |
| `viridis` | 黄→紫（色盲友好） | 推荐 |
| `coolwarm` | 蓝←→红 | 对比实验 |
| `plasma` | 黄→红→紫 | 深色背景 |
| **透明度 (alpha)** | | |
| `0.4` | 热图半透明，可看组织结构 | 默认 |
| `0.6` | 热图更明显 | 强调关注区域 |
| `0.2` | 热图较淡 | 背景信息重要时 |
| **热图层级 (vis_level)** | | |
| `-1` | 自动选择（最接近 32x 下采样）| 默认 |
| `0` | 最高分辨率 | 细节分析 |
| `1` | 4x 下采样 | 快速预览 |
| **重叠比例 (overlap)** | | |
| `0.0` | 无重叠 | 速度快 |
| `0.5` | 50% 重叠 | 默认，推荐 |
| `0.9` | 90% 重叠 | 精细热图 |

#### 6.4 命令

```bash
# 为 CLAM_SB 生成热图
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py \
    --save_exp_code dlbcl_clam_sb_heatmap \
    --config_file dlbcl_clam_sb.yaml

# 为 CLAM_MB 生成热图
CUDA_VISIBLE_DEVICES=1 python create_heatmaps.py \
    --save_exp_code dlbcl_clam_mb_heatmap \
    --config_file dlbcl_clam_mb.yaml

# 使用自定义重叠比例
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py \
    --save_exp_code dlbcl_clam_sb_heatmap \
    --overlap 0.5 \
    --config_file heatmaps/configs/dlbcl_clam_sb.yaml
```

#### 6.5 多组对照实验热图生成

为多组对照实验生成热图，建议创建多个配置文件：

```bash
# 1. CLAM_SB 热图配置
# heatmaps/configs/dlbcl_clam_sb.yaml
# 修改 ckpt_path: results/dlbcl_gcb_nongcb_clam_sb_s1/s_0_checkpoint.pt
# 修改 save_exp_code: dlbcl_clam_sb_heatmap

# 2. CLAM_MB 热图配置
# heatmaps/configs/dlbcl_clam_mb.yaml
# 修改 ckpt_path: results/dlbcl_gcb_nongcb_clam_mb_s1/s_0_checkpoint.pt
# 修改 save_exp_code: dlbcl_clam_mb_heatmap

# 3. MIL 基线热图配置
# heatmaps/configs/dlbcl_mil.yaml
# 修改 ckpt_path: results/dlbcl_gcb_nongcb_mil_s1/s_0_checkpoint.pt
# 修改 model_type: mil
# 修改 save_exp_code: dlbcl_mil_heatmap
```

然后依次运行：

```bash
# 生成各组热图
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file heatmaps/configs/dlbcl_clam_sb.yaml
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file heatmaps/configs/dlbcl_clam_mb.yaml
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file heatmaps/configs/dlbcl_mil.yaml
```

#### 6.6 输出文件

```
heatmaps/
├── heatmap_raw_results/              # 原始结果
│   └── dlbcl_clam_sb_heatmap/
│       └── GCB/
│           └── slide_001/
│               ├── slide_001_mask.jpg
│               ├── slide_001_blockmap.h5
│               └── ...
├── heatmap_production_results/       # 最终热图
│   └── dlbcl_clam_sb_heatmap/
│       └── GCB/
│           ├── slide_001_heatmap.jpg    # 热图叠加图像
│           ├── slide_001_orig.jpg       # 原始图像
│           └── ...
```

#### 6.7 参数速查表

| 参数类别 | 参数 | 说明 |
|----------|------|------|
| **实验** | `save_exp_code` | 输出目录名称 |
| **数据** | `data_dir` | WSI 文件目录 |
| | `process_list` | 处理列表 CSV |
| | `slide_ext` | 文件扩展名（.sdpc/.svs）|
| | `label_dict` | 标签映射 |
| **模型** | `ckpt_path` | checkpoint 路径 |
| | `model_type` | clam_sb/clam_mb/mil |
| | `embed_dim` | 特征维度 |
| **热图** | `cmap` | 颜色映射 |
| | `alpha` | 透明度 |
| | `vis_level` | 可视化层级 |
| | `overlap` | Patch 重叠比例 |

---

## 五、Task 配置与数据集

### 5.1 支持的任务

| Task | 说明 | 类别 | 标签 | CSV文件 |
|------|------|------|------|----------|
| `task_1_tumor_vs_normal` | 肿瘤 vs 正常 | 2 | `tumor_tissue`, `normal_tissue` | tumor_vs_normal_dummy_clean.csv |
| `task_2_tumor_subtyping` | 肿瘤亚型分型 | 3 | 多亚型标签 | tumor_subtyping_dummy_clean.csv |
| `task_3_dlbcl_coo` | DLBCL GCB vs non-GCB | 2 | `GCB`, `non-GCB` | gcb_vs_nongcb.csv |
| `task_4_tcga_dlbc` | TCGA DLBCL GCB vs non-GCB | 2 | `GCB`, `non-GCB` | tcga_dlbc_final_labels.csv |

### 5.2 数据集配置对应关系

在 `main.py` 中：

```python
if args.task == 'task_1_tumor_vs_normal':
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
        label_dict={'tumor_tissue': 0, 'normal_tissue': 1},
        patient_strat=True,
    )

elif args.task == 'task_2_tumor_subtyping':
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
        patient_strat=True,
        subtyping=True,
    )

elif args.task == 'task_3_dlbcl_coo':
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/gcb_vs_nongcb.csv',
        data_dir=os.path.join(args.data_root_dir, 'dlbcl_resnet_features'),
        label_dict={'GCB': 0, 'non-GCB': 1},
        patient_strat=False,  # 按 slide 级别划分
    )

elif args.task == 'task_4_tcga_dlbc':
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tcga_dlbc_final_labels.csv',
        data_dir=os.path.join(args.data_root_dir, 'tcga_dlbc_resnet_features'),
        label_dict={'GCB': 0, 'non-GCB': 1},
        patient_strat=True,
    )
```

### 5.3 如何添加新数据集

如果你有新数据集，需要完成以下步骤：

#### 步骤1: 准备CSV文件

在 `dataset_csv/` 目录下创建CSV文件，格式如下：

```csv
case_id,slide_id,label
patient_001,slide_001,GCB
patient_001,slide_002,non-GCB
patient_002,slide_003,GCB
...
```

#### 步骤2: 修改 main.py

在 `main.py` 中添加新的 task 分支（参考 task_4_tcga_dlbc 的格式）：

```python
elif args.task == 'task_5_your_new_task':
    args.n_classes = 2  # 根据你的类别数修改
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/your_new_csv.csv',
        data_dir=os.path.join(args.data_root_dir, 'your_features_folder'),
        label_dict={'label_1': 0, 'label_2': 1},
        patient_strat=True,  # 或 False，按需设置
        shuffle=False,
        seed=args.seed,
        print_info=True,
        ignore=[]
    )
```

#### 步骤3: 修改 eval.py（如需评估）

在 `eval.py` 中同样添加对应的分支。

#### 步骤4: 执行训练流程

```bash
# 1. Patch切割
python create_patches_fp.py --source raw_slides/your_data/ --save_dir results_yourdata/ --patch_size 256 --seg --patch

# 2. 特征提取
python extract_features_fp.py --data_h5_dir results_yourdata --data_slide_dir raw_slides/your_data --csv_path results_yourdata/process_list_autogen.csv --feat_dir features_yourdata --slide_ext .sdpc

# 3. 数据划分
python create_splits_seq.py --task task_5_your_new_task --seed 1 --k 10

# 4. 训练
python main.py --task task_5_your_new_task --data_root_dir features_yourdata --results_dir results_yourdata --exp_code your_exp_name --model_type clam_sb --k 10
```

---

## 六、常用命令速查表

### 6.1 快速开始（标准流程）

```bash
# 1. Patch 切割
python create_patches_fp.py --source raw_slides/ --save_dir results/ --patch_size 256 --seg --patch

# 2. 特征提取（ResNet-50）
python extract_features_fp.py --data_h5_dir results --data_slide_dir raw_slides --csv_path results/process_list_autogen.csv --feat_dir features --slide_ext .sdpc

# 3. 数据划分
python create_splits_seq.py --task task_3_dlbcl_coo --seed 1 --label_frac 1 --k 10

# 4. 训练（CLAM_SB）
python main.py --task task_3_dlbcl_coo --data_root_dir features --results_dir results --exp_code exp_001 --model_type clam_sb --k 10 --lr 2e-4 --drop_out 0.25 --bag_loss ce --inst_loss svm

# 5. 评估
python eval.py --models_exp_code exp_001_s1 --save_exp_code exp_001_eval --task task_3_dlbcl_coo --model_type clam_sb --data_root_dir features
```

### 6.2 不同模型对比

| 模型 | --model_type | --inst_loss | 特点 |
|------|--------------|-------------|------|
| CLAM_SB + WS | `clam_sb` | `svm`/`ce` | 单分支 + 弱监督（推荐）|
| CLAM_SB noWS | `clam_sb` | `svm`/`ce` + `--no_inst_cluster` | 单分支，无弱监督 |
| CLAM_MB | `clam_mb` | `svm`/`ce` | 多分支，适用于多分类 |
| MIL | `mil` | 无 | 标准基线 |

### 6.3 不同特征提取器

| 特征提取器 | --model_name | 性能 | 备注 |
|------------|--------------|------|------|
| ResNet-50 | `resnet50_trunc` | 基础 | 默认，无需额外设置 |
| UNI | `uni_v1` | **SOTA** | 需要申请权限 |
| CONCH | `conch_v1` | **SOTA** | 需要申请权限 |

---

## 七、注意事项

1. **数据泄露防范**: 按 `case_id`（患者）级别划分数据，避免同一患者的数据同时出现在训练集和测试集
2. **GPU 内存**: 如果 OOM，降低 `--batch_size` 或使用 `--model_size small`
3. **早停机制**: 建议启用 `--early_stopping`，防止过拟合
4. **SDPC 格式**: 使用 `--slide_ext .sdpc` 参数
5. **少样本学习**: 调整 `--label_frac` 参数（如 0.25 表示只用 25% 的标签训练）

---

## 八、多数据集训练指南

### 8.1 三个DLBCL数据集

本项目支持同时训练三个DLBCL数据集（GCB vs non-GCB二分类）：

| 数据集 | CSV文件 | 切片数 | WSI格式 | 特征目录 | 状态 |
|--------|---------|--------|---------|----------|------|
| nanchang_dlbcl | nanchang_dlbcl.csv | 389 | .sdpc | features/nanchang_resnet_features/ | 可训练 |
| tcga_dlbcl | tcga_dlbcl.csv | 39 | .svs | features/tcga_resnet_features/ | 可训练 |
| dlbcl_morph | dlbcl_morph.csv | 185 | .svs | features/morph_resnet_features/ | 需提取特征 |

> **注意**: 三个数据集共用 `task_3_dlbcl_coo`，通过 `--dataset` 参数选择不同数据集。

### 8.2 数据准备与特征提取

#### 8.2.1 nanchang_dlbcl (已就绪)

```bash
# 特征已提取，可直接训练
# 如需重新提取（使用ResNet-50）：
python create_patches_fp.py --source raw_slides/nanchang_dlbcl/ --save_dir results/nanchang_dlbcl/ --patch_size 256 --seg --patch --stitch
python extract_features_fp.py --data_h5_dir results/nanchang_dlbcl/ --data_slide_dir raw_slides/nanchang_dlbcl/ --csv_path dataset_csv/nanchang_dlbcl.csv --feat_dir features/nanchang_resnet_features/ --batch_size 512 --slide_ext .sdpc

# 如需使用UNI模型：
export UNI_CKPT_PATH=/path/to/uni_checkpoint.pth
python extract_features_fp.py ... --model_name uni_v1 --feat_dir features/nanchang_uni_features/
```

#### 8.2.2 tcga_dlbcl (已就绪)

```bash
# 特征已提取，可直接训练
# 如需重新提取：
python create_patches_fp.py --source raw_slides/TCGA-DLBC/ --save_dir results/tcga_dlbcl/ --patch_size 256 --seg --patch --stitch
python extract_features_fp.py --data_h5_dir results/tcga_dlbcl/ --data_slide_dir raw_slides/TCGA-DLBC/ --csv_path dataset_csv/tcga_dlbcl.csv --feat_dir features/tcga_resnet_features/ --batch_size 512 --slide_ext .svs
```

#### 8.2.3 dlbcl_morph (需提取特征)

```bash
# 1. Patch提取
python create_patches_fp.py --source raw_slides/DLBCL-Morphology/ --save_dir results/dlbcl_morph/ --patch_size 256 --seg --patch --stitch

# 2. 特征提取 (ResNet-50)
python extract_features_fp.py --data_h5_dir results/dlbcl_morph/ --data_slide_dir raw_slides/DLBCL-Morphology/ --csv_path dataset_csv/dlbcl_morph.csv --feat_dir features/morph_resnet_features/ --batch_size 512 --slide_ext .svs
```

### 8.3 数据划分与训练

> **注意**: main.py 已修改支持 `--dataset` 参数（nanchang/tcga/morph/all），可直接使用。

#### 8.3.1 创建数据划分

```bash
# 为各数据集创建数据划分（使用不同seed确保独立性）
python create_splits_seq.py --task task_3_dlbcl_coo --seed 1 --k 10  # nanchang
python create_splits_seq.py --task task_3_dlbcl_coo --seed 2 --k 10  # tcga
python create_splits_seq.py --task task_3_dlbcl_coo --seed 3 --k 10  # morph
```

```python
# 在 argument 中添加：
parser.add_argument('--dataset', type=str, default='nanchang',
                    choices=['nanchang', 'tcga', 'morph'],
                    help='选择数据集')

# 修改 task_3_dlbcl_coo 分支：
elif args.task == 'task_3_dlbcl_coo':
    args.n_classes = 2
    # 根据 --dataset 参数选择CSV和特征目录
    if args.dataset == 'nanchang':
        csv_path = 'dataset_csv/nanchang_dlbcl.csv'
        data_dir = os.path.join(args.data_root_dir, 'nanchang_resnet_features')
    elif args.dataset == 'tcga':
        csv_path = 'dataset_csv/tcga_dlbcl.csv'
        data_dir = os.path.join(args.data_root_dir, 'tcga_resnet_features')
    elif args.dataset == 'morph':
        csv_path = 'dataset_csv/dlbcl_morph.csv'
        data_dir = os.path.join(args.data_root_dir, 'morph_resnet_features')

    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=data_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'GCB': 0, 'non-GCB': 1},
        patient_strat=True,
        ignore=[]
    )
```

#### 8.3.2 创建各数据集的数据划分

```bash
# 方法1: 使用 --split_dir 指定不同的划分目录
# 先创建默认划分（基于 nanchang）
python create_splits_seq.py --task task_3_dlbcl_coo --seed 1 --label_frac 1 --k 10

# 复制为各数据集的划分（需要根据各数据集的slide重新生成）
# 由于create_splits_seq.py使用固定的CSV，需要手动或修改代码

# 方法2: 使用不同seed创建多套划分（简单方案）
python create_splits_seq.py --task task_3_dlbcl_coo --seed 1 --k 10  # nanchang用
python create_splits_seq.py --task task_3_dlbcl_coo --seed 2 --k 10  # tcga用
python create_splits_seq.py --task task_3_dlbcl_coo --seed 3 --k 10  # morph用

# 然后通过 --split_dir 指定
# nanchang: splits/task_3_dlbcl_coo_100 (seed=1)
# tcga: splits/task_3_dlbcl_coo_100 (seed=2)
# morph: splits/task_3_dlbcl_coo_100 (seed=3)
```

#### 8.3.3 使用不同数据集训练

```bash
# 训练 nanchang_dlbcl
python main.py --task task_3_dlbcl_coo --dataset nanchang \
    --data_root_dir features \
    --exp_code nanchang_clam_sb \
    --split_dir splits/task_3_dlbcl_coo_100 \
    ...

# 训练 tcga_dlbcl
python main.py --task task_3_dlbcl_coo --dataset tcga \
    --data_root_dir features \
    --exp_code tcga_clam_sb \
    --split_dir splits/task_3_dlbcl_coo_100 \
    ...

# 训练 dlbcl_morph
python main.py --task task_3_dlbcl_coo --dataset morph \
    --data_root_dir features \
    --exp_code morph_clam_sb \
    --split_dir splits/task_3_dlbcl_coo_100 \
    ...
```

### 8.4 分开训练

#### 8.4.1 训练 nanchang_dlbcl

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir features \
    --results_dir results/nanchang_dlbcl \
    --exp_code nanchang_clam_sb \
    --model_type clam_sb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --log_data \
    --embed_dim 1024 \
    --weighted_sample \
    --split_dir splits/task_3_dlbcl_coo_100
```

#### 8.4.2 训练 tcga_dlbcl

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir features \
    --results_dir results/tcga_dlbcl \
    --exp_code tcga_clam_sb \
    --model_type clam_sb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --log_data \
    --embed_dim 1024 \
    --weighted_sample \
    --split_dir splits/task_3_dlbcl_coo_100
```

#### 8.4.3 训练 dlbcl_morph

```bash
# 先确保特征已提取
CUDA_VISIBLE_DEVICES=2 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir features \
    --results_dir results/dlbcl_morph \
    --exp_code morph_clam_sb \
    --model_type clam_sb \
    --model_size small \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 10 \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --log_data \
    --embed_dim 1024 \
    --weighted_sample \
    --split_dir splits/task_3_dlbcl_coo_100
```

### 8.5 整合训练（合并数据集）

如果希望将三个数据集合并为一个更大的数据集训练：

#### 8.5.1 合并CSV

```bash
# 合并三个数据集的CSV（需确保case_id和slide_id不冲突）
python3 -c "
import pandas as pd
df1 = pd.read_csv('dataset_csv/nanchang_dlbcl.csv')
df2 = pd.read_csv('dataset_csv/tcga_dlbcl.csv')
df3 = pd.read_csv('dataset_csv/dlbcl_morph.csv')
# 为避免ID冲突，可以添加前缀
df1['case_id'] = 'nanchang_' + df1['case_id'].astype(str)
df1['slide_id'] = 'nanchang_' + df1['slide_id'].astype(str)
df2['case_id'] = 'tcga_' + df2['case_id'].astype(str)
df2['slide_id'] = 'tcga_' + df2['slide_id'].astype(str)
df3['case_id'] = 'morph_' + df3['case_id'].astype(str)
df3['slide_id'] = 'morph_' + df3['slide_id'].astype(str)
df = pd.concat([df1, df2, df3], ignore_index=True)
df.to_csv('dataset_csv/dlbcl_all.csv', index=False)
print(f'合并后: {len(df)} 切片, {df[\"case_id\"].nunique()} 患者')
print(f'GCB: {(df[\"label\"]==\"GCB\").sum()}, non-GCB: {(df[\"label\"]==\"non-GCB\").sum()}')
"
```

#### 8.5.2 合并特征

```bash
# 复制特征文件到统一目录
mkdir -p features/all_resnet_features
cp features/nanchang_resnet_features/pt_files/* features/all_resnet_features/
cp features/tcga_resnet_features/pt_files/* features/all_resnet_features/
# morph的特征（提取后）
cp features/morph_resnet_features/pt_files/* features/all_resnet_features/
```

#### 8.5.3 创建数据划分并训练

```bash
# 创建数据划分（需要修改main.py支持新CSV，或直接用合并后的CSV）
# 训练合并后的数据集
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --dataset all \
    --data_root_dir features \
    --exp_code all_clam_sb \
    --results_dir results/dlbcl_all \
    --exp_code dlbcl_all_clam_sb \
    --model_type clam_sb \
    --k 10 \
    ...
```

### 8.6 评估

分别评估三个数据集：

```bash
# nanchang
python eval.py --models_exp_code nanchang_clam_sb_s1 --save_exp_code nanchang_eval --task task_3_dlbcl_coo --model_type clam_sb --data_root_dir features/dlbcl_resnet_features

# tcga
python eval.py --models_exp_code tcga_clam_sb_s1 --save_exp_code tcga_eval --task task_3_dlbcl_coo --model_type clam_sb --data_root_dir features/TCGA-DLBC

# morph
python eval.py --models_exp_code morph_clam_sb_s1 --save_exp_code morph_eval --task task_3_dlbcl_coo --model_type clam_sb --data_root_dir features/dlbcl_morph
```

### 8.7 并行训练

```bash
# 三个数据集并行训练（后台运行）
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --task task_3_dlbcl_coo --data_root_dir features/dlbcl_resnet_features ...' > nanchang.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --task task_3_dlbcl_coo --data_root_dir features/TCGA-DLBC ...' > tcga.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --task task_3_dlbcl_coo --data_root_dir features/dlbcl_morph ...' > morph.log 2>&1 &
```

本项目支持同时训练和评估多个数据集。每个数据集需要独立的：
- CSV文件 (`dataset_csv/`)
- WSI文件目录 (`raw_slides/<dataset_name>/`)
- Patch结果目录 (`results_<dataset_name>/`)
- 特征目录 (`features_<dataset_name>/`)
- Task配置 (在 main.py 和 eval.py 中)

### 8.2 多数据集训练示例

#### 示例：同时训练 DLBCL 和 TCGA 数据集

```bash
# ============ 数据集1: DLBCL ============
# 1. Patch切割
python create_patches_fp.py --source raw_slides/ --save_dir results_dlbcl/ --patch_size 256 --seg --patch

# 2. 特征提取
python extract_features_fp.py --data_h5_dir results_dlbcl --data_slide_dir raw_slides --csv_path results_dlbcl/process_list_autogen.csv --feat_dir features_dlbcl --slide_ext .sdpc

# 3. 数据划分
python create_splits_seq.py --task task_3_dlbcl_coo --seed 1 --k 10

# 4. 训练
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo \
    --data_root_dir features_dlbcl \
    --results_dir results_dlbcl \
    --exp_code dlbcl_clam_sb \
    --model_type clam_sb \
    --k 10

# 5. 评估
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --task task_3_dlbcl_coo \
    --models_exp_code dlbcl_clam_sb_s1 \
    --save_exp_code dlbcl_clam_sb_eval \
    --model_type clam_sb \
    --data_root_dir features_dlbcl \
    --results_dir results_dlbcl

# ============ 数据集2: TCGA ============
# 1. Patch切割
python create_patches_fp.py --source raw_slides/TCGA/ --save_dir results_tcga/ --patch_size 256 --seg --patch

# 2. 特征提取
python extract_features_fp.py --data_h5_dir results_tcga --data_slide_dir raw_slides/TCGA --csv_path results_tcga/process_list_autogen.csv --feat_dir features_tcga --slide_ext .sdpc

# 3. 数据划分
python create_splits_seq.py --task task_4_tcga_dlbc --seed 1 --k 10

# 4. 训练
CUDA_VISIBLE_DEVICES=1 python main.py \
    --task task_4_tcga_dlbc \
    --data_root_dir features_tcga \
    --results_dir results_tcga \
    --exp_code tcga_clam_sb \
    --model_type clam_sb \
    --k 10
```

### 8.3 关键参数区分

| 参数 | 用途 | 示例 |
|------|------|------|
| `--task` | 区分数据集任务 | `task_3_dlbcl_coo`, `task_4_tcga_dlbc` |
| `--exp_code` | 区分实验名称 | `dlbcl_clam_sb`, `tcga_clam_sb` |
| `--results_dir` | 区分输出目录 | `results_dlbcl`, `results_tcga` |
| `--data_root_dir` | 区分特征目录 | `features_dlbcl`, `features_tcga` |

### 8.4 并行训练

如果有多个GPU，可以并行训练不同数据集：

```bash
# 并行训练（后台运行）
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --task task_3_dlbcl_coo ...' > dlbcl.log 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --task task_4_tcga_dlbc ...' > tcga.log 2>&1 &
```

### 8.5 迁移学习（模型继承训练）

CLAM默认不支持直接加载预训练权重进行微调，但可以通过以下方式实现迁移学习。

#### 8.5.1 方法一：手动加载预训练权重

修改 `utils/core_utils.py` 中的训练代码，在模型创建后加载预训练权重：

```python
# 在 utils/core_utils.py 的 train 函数中，找到 model 创建的位置（约第170-200行）
# 在 return dataset 之前添加加载代码：

# 示例：加载预训练权重
pretrained_path = '/path/to/pretrained/checkpoint.pt'
if os.path.exists(pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    # 过滤掉不匹配的层（如分类头）
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)} layers from pretrained model")
```

#### 8.5.2 方法二：修改 main.py 添加 --resume 参数

在 `main.py` 中添加 resume 功能：

```python
# 在 argparse 部分添加：
parser.add_argument('--resume', type=str, default=None,
                    help='path to pretrained checkpoint for transfer learning')

# 在训练前加载权重（大约在第70行附近）：
if args.resume:
    print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    print("Pretrained weights loaded successfully!")
```

然后可以使用：

```bash
# 先在数据集1上预训练
python main.py --task task_3_dlbcl_coo --exp_code dlbcl_pretrain ...

# 加载预训练权重，在数据集2上微调
python main.py --task task_4_tcga_dlbc --exp_code tcga_finetune \
    --resume results/dlbcl_pretrain_s1/s_0_checkpoint.pt ...
```

#### 8.5.3 迁移学习训练策略

**全量微调（Fine-tuning）**：
```bash
# 加载预训练权重，用较低学习率微调
python main.py --task task_4_tcga_dlbc \
    --resume results/dlbcl_pretrain_s1/s_0_checkpoint.pt \
    --lr 1e-5 \  # 使用更小的学习率
    --drop_out 0.5  # 适当增大dropout防止过拟合
    ...
```

**特征提取（Frozen）**：
```bash
# 冻结特征提取层，只训练分类头
# 需要修改代码，在 model 创建后冻结特征层：
for param in model.feature_extractor.parameters():
    param.requires_grad = False

# 然后用较高学习率只训练分类头
python main.py --task task_4_tcga_dlbc \
    --resume results/dlbcl_pretrain_s1/s_0_checkpoint.pt \
    --lr 1e-3 ...
```

#### 8.5.4 迁移学习注意事项

1. **特征维度匹配**：预训练模型和目标任务的特征维度需一致（`--embed_dim` 相同）
2. **类别数变化**：如果类别数不同，分类层权重不能直接加载
3. **学习率设置**：
   - 全量微调：使用原学习率的 1/10（如 2e-5）
   - 只训练分类头：可以使用原学习率
4. **数据量差异**：目标数据集数据量少时，建议用较小的学习率和较大的dropout
5. **早停策略**：迁移学习更容易过拟合，建议启用早停并设置较小的patience

---

*本文档最后更新于 2026-03-23*
