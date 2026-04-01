# CLAM 项目路径总结

## 根目录
```
/home/shanyiye/CLAM/
```

---

## 主要目录

| 目录 | 路径 | 说明 |
|---|---|---|
| 项目根目录 | `/home/shanyiye/CLAM/` | CLAM 主目录 |
| 原始切片 | `/home/shanyiye/CLAM/raw_slides/` | WSI 原始文件 (.svs/.tiff) |
| 数据集CSV | `/home/shanyiye/CLAM/dataset_csv/` | 数据标注文件 |
| 特征存储 | `/home/shanyiye/CLAM/features/` | 提取的特征 (.pt) |
| 热力图输出 | `/home/shanyiye/CLAM/heatmaps/` | 可视化热力图 |
| 训练结果 | `/home/shanyiye/CLAM/results/` | 模型训练输出 |
| 评估结果 | `/home/shanyiye/CLAM/eval_results/` | 模型评估输出 |
| 分析结果 | `/home/shanyiye/CLAM/analysis_results/` | 模型分析结果 |
| 数据划分 | `/home/shanyiye/CLAM/splits/` | K-Fold 交叉验证划分 |
| 预设配置 | `/home/shanyiye/CLAM/presets/` | 预设数据集配置 |

---

## 核心脚本

| 脚本 | 路径 | 功能 |
|---|---|---|
| 主训练脚本 | `/home/shanyiye/CLAM/main.py` | K-Fold 交叉验证训练 |
| 评估脚本 | `/home/shanyiye/CLAM/eval.py` | 模型评估 |
| 分析脚本 | `/home/shanyiye/CLAM/analyze_eval_results.py` | 分析评估结果 |
| 特征提取 | `/home/shanyiye/CLAM/extract_features.py` | 从WSI提取特征 |
| 创建补丁 | `/home/shanyiye/CLAM/create_patches.py` | 切割WSI生成patch |
| 生成热力图 | `/home/shanyiye/CLAM/create_heatmaps.py` | 注意力热图可视化 |
| 创建数据划分 | `/home/shanyiye/CLAM/create_splits_seq.py` | 生成K-Fold划分 |
| CSV转换 | `/home/shanyiye/CLAM/make_clam_csv.py` | 生成CLAM格式CSV |

---

## 代码模块

### models/ - 模型定义
```
/home/shanyiye/CLAM/models/
├── builder.py           # 模型构建器
├── model_clam.py       # CLAM_SB/CLAM_MB 模型
├── model_mil.py        # 标准 MIL 模型
├── timm_wrapper.py     # TIMM 包装器
└── resnet_custom_dep.py # 自定义 ResNet
```

### dataset_modules/ - 数据集模块
```
/home/shanyiye/CLAM/dataset_modules/
├── dataset_generic.py   # 通用数据集类
├── dataset_h5.py       # HDF5 数据集
└── wsi_dataset.py      # WSI 数据集
```

### utils/ - 工具函数
```
/home/shanyiye/CLAM/utils/
├── core_utils.py       # 训练/验证循环
├── eval_utils.py       # 评估工具
├── file_utils.py       # 文件操作
├── transform_utils.py   # 数据变换
├── utils.py            # 通用工具
└── constants.py       # 常量定义
```

### wsi_core/ - WSI处理核心
```
/home/shanyiye/CLAM/wsi_core/
├── WholeSlideImage.py    # WSI 处理类
├── batch_process_utils.py # 批处理
├── wsi_utils.py          # WSI 工具
└── util_classes.py      # 工具类
```

---

## 数据集 CSV 路径

| 文件 | 路径 |
|---|---|
| 肿瘤分型 | `/home/shanyiye/CLAM/dataset_csv/tumor_subtyping_dummy_clean.csv` |
| 肿瘤vs正常 | `/home/shanyiye/CLAM/dataset_csv/tumor_vs_normal_dummy_clean.csv` |
| 二分型(xlsx) | `/home/shanyiye/CLAM/dataset_csv/二分型.xlsx` |
| 数据部分1 | `/home/shanyiye/CLAM/dataset_csv/csv_part1.csv` |
| 数据部分2 | `/home/shanyiye/CLAM/dataset_csv/csv_part2.csv` |

---

## 训练结果路径

### 任务1: Tumor vs Normal
```
/home/shanyiye/CLAM/splits/task_1_tumor_vs_normal_75/
```

### 任务2: Tumor Subtyping
```
/home/shanyiye/CLAM/splits/task_2_tumor_subtyping_50/
```

### 任务3: DLBCL GCB vs non-GCB
```
/home/shanyiye/CLAM/splits/task_3_dlbcl_coo_100/
```

### 训练结果 (results/)

| 实验 | 路径 |
|---|---|
| CLAM_SB 训练 | `/home/shanyiye/CLAM/results/dlbcl_gcb_nongcb_clam_sb_s1/` |
| MIL 训练 | `/home/shanyiye/CLAM/results/dlbcl_gcb_nongcb_mil_s1/` |
| 补丁文件 | `/home/shanyiye/CLAM/results/patches/*.h5` |
| 组织掩膜 | `/home/shanyiye/CLAM/results/masks/*.jpg` |

---

## 评估结果路径 (eval_results/)

| 实验 | 路径 |
|---|---|
| CLAM_SB 评估 | `/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_eval/` |
| CLAM_SB (NoWS) 评估 | `/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_nows_eval/` |
| MIL 评估 | `/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_gcb_nongcb_mil_eval/` |

每个评估目录包含：
- `summary.csv` - 总体评估摘要
- `fold_0.csv` ~ `fold_9.csv` - 各折评估结果
- `eval_experiment_*.txt` - 评估实验配置

---

## 分析结果路径 (analysis_results/)

```
/home/shanyiye/CLAM/analysis_results/
├── model_comparison.csv          # 模型对比结果
├── CLAM_SB_WS/                   # CLAM_SB 全采样分析
│   ├── report.txt
│   ├── case_confusion_matrix.png
│   ├── slide_confusion_matrix.png
│   ├── case_roc.png
│   ├── slide_roc.png
│   ├── borderline_cases.csv
│   ├── wrong_cases.csv
│   ├── wrong_slides.csv
│   ├── all_case_predictions.csv
│   └── all_slide_predictions.csv
└── CLAM_SB_noWS/                 # CLAM_SB 无采样分析
    ├── report.txt
    ├── case_confusion_matrix.png
    ├── slide_confusion_matrix.png
    ├── case_roc.png
    ├── slide_roc.png
    ├── borderline_cases.csv
    ├── wrong_cases.csv
    ├── wrong_slides.csv
    ├── all_case_predictions.csv
    └── all_slide_predictions.csv
```

---

## 中间结果路径

| 类型 | 路径 |
|---|---|
| 补丁文件 | `/home/shanyiye/CLAM/results/patches/*.h5` |
| 组织掩膜 | `/home/shanyiye/CLAM/results/masks/*.jpg` |
| 进程列表 | `/home/shanyiye/CLAM/results/process_list_autogen.csv` |

---

*最后更新：2026-03-12*
