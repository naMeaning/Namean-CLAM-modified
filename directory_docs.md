# CLAM 项目目录说明文档

> 本文档为 AI 上下文文档，用于帮助 AI 理解 CLAM 项目的结构和功能

---

## 一、项目概述

**CLAM** (Clustering-constrained Attention Multiple Instance Learning) 是一个用于全切片病理图像 (Whole Slide Image, WSI) 分类的可解释弱监督学习框架。

- **论文**: *Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images* (Lu et al., Nature Biomedical Engineering, 2021)
- **核心功能**: 在仅有 slide 级别标签（弱监督）的条件下，对 WSI 进行分类
- **支持的任务**: 二分类（肿瘤 vs 正常）、多分类（肿瘤亚型分型）

---

## 二、顶层目录结构

```
/home/shanyiye/CLAM/
├── main.py                  # 训练主入口
├── eval.py                  # 评估主入口
├── create_patches.py        # Patch 切割（单线程）
├── create_patches_fp.py     # Patch 切割（多进程版本）
├── extract_features.py      # 特征提取
├── extract_features_fp.py   # 特征提取（多进程版本）
├── create_heatmaps.py       # 生成注意力热图
├── create_splits_seq.py     # 生成 K-Fold 数据划分
├── make_clam_csv.py         # 生成 CLAM 格式 CSV
├── analyze_eval_results.py  # 分析评估结果
├── build_preset.py          # 构建预设配置
├── env.yml                  # Conda 环境配置
├── README.md                # 项目说明
├── LICENSE.md               # 许可证
├── project_summary.md       # 项目总结文档
├── dir_summary.md           # 目录摘要
│
├── models/                  # 模型定义
├── utils/                   # 工具函数
├── dataset_modules/         # 数据集加载
├── wsi_core/               # WSI 处理核心
├── vis_utils/              # 可视化工具
├── presets/                # 预设配置
├── docs/                   # 文档和图片资源
│
├── raw_slides/             # 原始 WSI 文件存储
├── features/               # 提取的特征存储
│   └── dlbcl_resnet_features/
├── dataset_csv/             # 数据集 CSV 文件
│   ├── tumor_vs_normal_dummy_clean.csv
│   ├── tumor_subtyping_dummy_clean.csv
│   ├── gcb_vs_nongcb.csv
│   └── gcb_vs_nongcb_for_features.csv
├── splits/                 # K-Fold 数据划分
│   ├── task_1_tumor_vs_normal_75/
│   ├── task_2_tumor_subtyping_50/
│   └── task_3_dlbcl_coo_100/
├── results/                # 训练结果和中间产物
│   ├── patches/            # 切割的 patch 图像
│   ├── masks/              # 组织掩膜
│   ├── stitches/           # patch 拼接图
│   ├── dlbcl_gcb_nongcb_clam_sb_s1/
│   ├── dlbcl_gcb_nongcb_clam_sb_nows_s1/
│   └── dlbcl_gcb_nongcb_mil_s1/
├── heatmaps/               # 热图生成相关
│   ├── configs/
│   ├── demo/
│   └── process_lists/
├── eval_results/           # 评估结果
│   ├── EVAL_dlbcl_gcb_nongcb_clam_sb_eval/
│   ├── EVAL_dlbcl_gcb_nongcb_clam_sb_nows_eval/
│   └── EVAL_dlbcl_gcb_nongcb_mil_eval/
├── analysis_results/       # 分析结果
│   ├── MIL/
│   ├── CLAM_SB_WS/
│   ├── CLAM_SB_noWS/
│   └── model_comparison.csv
├── .agent/                 # Agent 技能配置
├── .agents/                # Agent 规则配置
└── __pycache__/            # Python 缓存
```

---

## 三、核心模块详解

### 3.1 `models/` - 模型定义

| 文件 | 功能说明 |
|------|----------|
| `model_clam.py` | CLAM 模型实现（单分支 CLAM_SB、多分支 CLAM_MB） |
| `model_mil.py` | 标准 MIL 模型（MIL_fc, MIL_fc_mc） |
| `builder.py` | 模型构建器，支持多种预训练特征提取器 |
| `resnet_custom_dep.py` | 自定义 ResNet 实现 |
| `timm_wrapper.py` | timm 库包装器 |

**支持的特征提取模型**:
- `resnet50_trunc` - ResNet50 截断版
- `uni_v1` - UNI 病理大模型
- `conch_v1` / `conch_v1_5` - CONCH 病理大模型

### 3.2 `utils/` - 工具函数

| 文件 | 功能说明 |
|------|----------|
| `core_utils.py` | 训练/验证/测试循环核心逻辑，包含 EarlyStopping、Accuracy_Logger |
| `utils.py` | 通用工具：数据划分生成、数据加载器构建、优化器创建 |
| `eval_utils.py` | 独立评估工具：模型加载、评估执行、结果汇总 |
| `file_utils.py` | 文件操作工具 |
| `transform_utils.py` | 数据变换工具 |
| `constants.py` | 常量定义 |

### 3.3 `dataset_modules/` - 数据集加载

| 文件 | 功能说明 |
|------|----------|
| `dataset_generic.py` | 通用 MIL 数据集类 Generic_MIL_Dataset |
| `dataset_h5.py` | H5 格式数据集加载器 |
| `wsi_dataset.py` | WSI 数据集实现 |

### 3.4 `wsi_core/` - WSI 处理核心

| 文件 | 功能说明 |
|------|----------|
| `WholeSlideImage.py` | WSI 读取、处理、热图生成核心类 |
| `wsi_utils.py` | WSI 工具函数 |
| `util_classes.py` | 实用类定义（Annotation, Contour 等） |
| `batch_process_utils.py` | 批量处理工具 |

**主要功能**:
- 组织分割 (segmentTissue)
- Patch 坐标提取 (process_contour)
- 注意力热图可视化 (visHeatmap)

### 3.5 `vis_utils/` - 可视化工具

| 文件 | 功能说明 |
|------|----------|
| `heatmap_utils.py` | 热图生成工具 |

### 3.6 `presets/` - 预设配置

| 文件 | 功能说明 |
|------|----------|
| `bwh_biopsy.csv` | BWH 活检预设 |
| `bwh_resection.csv` | BWH 切除术预设 |
| `tcga.csv` | TCGA 预设 |

### 3.7 `docs/` - 文档资源

包含项目文档和论文图片资源。

---

## 四、数据流程

```
WSI 文件 (.svs / .sdpc / .tiff)
    │
    ▼ [create_patches.py / create_patches_fp.py]
组织切割（segmentTissue）→ Patch 坐标提取（process_contour）
    │ 输出：results/patches/, results/masks/
    │
    ▼ [extract_features.py / extract_features_fp.py]
预训练编码器推理（ResNet-50 / UNI / CONCH）
    │ 输出：features/*.pt (特征), *.h5 (坐标)
    │
    ▼ [create_splits_seq.py]
K-Fold 数据划分
    │ 输出：splits/<task>_<label_frac>/splits_*.csv
    │
    ▼ [main.py]
CLAM / MIL 训练（K-Fold 交叉验证）
    │ 输出：results/<exp_code>_s<seed>/
    │
    ▼ [eval.py]
独立评估
    │ 输出：eval_results/EVAL_*/
    │
    ▼ [create_heatmaps.py]
注意力热图可视化
```

---

## 五、关键入口文件

### 5.1 `main.py` - 训练入口

```bash
python main.py \
  --task task_3_dlbcl_coo \
  --data_root_dir /home/shanyiye/CLAM/features \
  --results_dir /home/shanyiye/CLAM/results \
  --exp_code dlbcl_gcb_nongcb_clam_sb \
  --model_type clam_sb \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --bag_loss ce \
  --inst_loss svm \
  --log_data \
  --embed_dim 1024
```

### 5.2 `eval.py` - 评估入口

```bash
python eval.py \
  --k 10 \
  --models_exp_code dlbcl_gcb_nongcb_clam_sb_s1 \
  --save_exp_code dlbcl_gcb_nongcb_clam_sb_eval \
  --task task_3_dlbcl_coo \
  --model_type clam_sb \
  --results_dir /home/shanyiye/CLAM/results \
  --data_root_dir /home/shanyiye/CLAM/features \
  --embed_dim 1024 \
  --seed 1
```

### 5.3 `create_patches_fp.py` - Patch 切割

```bash
python create_patches_fp.py \
  --source raw_slides/ \
  --save_dir results/ \
  --patch_size 256 \
  --seg \
  --patch
```

### 5.4 `extract_features_fp.py` - 特征提取

```bash
python extract_features_fp.py \
  --data_h5_dir results \
  --data_slide_dir raw_slides \
  --csv_path results/csv_part1.csv \
  --feat_dir features \
  --batch_size 512 \
  --slide_ext .sdpc
```

### 5.5 `create_splits_seq.py` - 数据划分

```bash
python create_splits_seq.py \
  --task task_3_dlbcl_coo \
  --seed 1 \
  --label_frac 1 \
  --k 10
```

---

## 六、数据格式

### 6.1 CSV 数据集格式

| 列名 | 说明 |
|------|------|
| `case_id` | 患者 ID（用于 patient-level 划分） |
| `slide_id` | 切片 ID（文件名，不含扩展名） |
| `label` | 字符串标签（如 `tumor_tissue`, `normal_tissue`, `GCB`, `non-GCB`） |

### 6.2 特征文件格式

- `.pt` 文件: PyTorch tensor，形状为 `(N, D)`，N=patch数，D=特征维度（默认1024）
- `.h5` 文件: 包含 patch 坐标信息

### 6.3 训练结果目录结构

```
results/<exp_code>_s<seed>/
├── logs/                    # 训练日志
├── checkpoints/            # 模型检查点
│   └── s<fold>/<epoch>.pt
├── results.csv             # 训练结果汇总
├── splitted_id.csv         # 数据划分信息
└── ...（其他输出）
```

---

## 七、模型架构

### 7.1 CLAM_SB (Single Branch)

- 单个注意力分支
- 适用于二分类任务
- 支持 instance-level 聚类损失

### 7.2 CLAM_MB (Multi Branch)

- 多个注意力分支（每类一个）
- 适用于多分类任务
- 更细粒度的注意力可视化

### 7.3 MIL (Standard)

- 标准 MIL 网络
- 基础 baseline 模型
- 无注意力可视化

---

## 八、常用命令速查

| 操作 | 命令 |
|------|------|
| 创建 conda 环境 | `conda env create -f env.yml` |
| Patch 切割 | `python create_patches_fp.py --source raw_slides/ --save_dir results/` |
| 特征提取 | `python extract_features_fp.py --data_h5_dir results --feat_dir features` |
| 生成数据划分 | `python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1` |
| 训练 CLAM | `python main.py --model_type clam_sb --task task_1_tumor_vs_normal` |
| 评估模型 | `python eval.py --models_exp_code <exp_code> --model_type clam_sb` |
| 生成热图 | `python create_heatmaps.py --results_dir results --heatmap_dir heatmaps` |

---

## 九、注意事项

1. **WSI 格式支持**: 支持 .svs, .sdpc, .tiff 等格式，需要安装 openslide
2. **GPU 要求**: 特征提取和训练需要 CUDA 支持
3. **特征提取器**: 使用 UNI/CONCH 需要额外设置环境变量 `UNI_CKPT_PATH`、`CONCH_CKPT_PATH`
4. **数据划分**: 按 patient (case_id) 级别划分，避免数据泄露

---

*本文档最后更新于 2026-03-14*
