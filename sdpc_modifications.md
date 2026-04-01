# CLAM 项目 SDPC 格式支持与 Task 配置修改文档

> 本文档记录了 CLAM 项目针对 `.sdpc` 格式 WSI 文件的修改，以及新增的 `task_3_dlbcl_coo` 任务配置。
> 作为 AI 上下文档，用于理解项目的定制化内容。

---

## 一、SDPC 格式支持

### 1.1 背景

原始 CLAM 项目仅支持 `.svs`、`.tiff`、`.ndpi` 等标准 WSI 格式，使用 `openslide` 库加载。本项目为支持国产扫描仪生成的 `.sdpc` 格式文件，引入了 `opensdpc` 库，并对相关代码进行了修改。

### 1.2 修改的文件

| 文件路径 | 修改内容 |
|----------|----------|
| `wsi_core/WholeSlideImage.py` | WSI 加载逻辑、read_region 调用方式 |
| `extract_features_fp.py` | 特征提取时的 WSI 加载逻辑 |

### 1.3 核心修改详解

#### 1.3.1 WholeSlideImage.py - WSI 加载

**位置**: `wsi_core/WholeSlideImage.py` 第 46-52 行

```python
# --- 修改开始 ---
if path.endswith('.sdpc'):
    # 使用 opensdpc 加载 sdpc 文件
    self.wsi = opensdpc.OpenSdpc(path)
else:
    # 其他格式（.svs, .tiff 等）依然使用 openslide
    self.wsi = openslide.open_slide(path)
# --- 修改结束 ---
```

**说明**:
- 检测文件后缀名是否为 `.sdpc`
- 如果是，使用 `opensdpc.OpenSdpc()` 加载
- 否则使用标准的 `openslide.open_slide()` 加载

#### 1.3.2 WholeSlideImage.py - read_region 调用

**位置**: `wsi_core/WholeSlideImage.py` 多处

由于 `opensdpc` 和 `openslide` 的 API 略有不同，需要注意以下几点：

| 场景 | 代码位置 | openslide 兼容处理 |
|------|----------|-------------------|
| 组织分割读取 | 第 169 行 | `self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level])` |
| 可视化读取 | 第 224 行 | `.convert("RGB")` 转换 |
| Patch 提取 | 第 351 行 | `.convert("RGB")` 转换 |
| 热图生成 | 第 664、768 行 | `.convert("RGB")` 转换 |

**关键差异**:
- `opensdpc.OpenSdpc.read_region()` 返回的对象需要调用 `.convert("RGB")` 才能转为 PIL Image
- `openslide` 的 `read_region` 返回的对象本身已是 RGB 模式，但仍需显式调用以保持兼容

```python
# 示例：Patch 提取
patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
```

#### 1.3.3 extract_features_fp.py - 特征提取时的加载

**位置**: `extract_features_fp.py` 第 119-127 行

```python
# 获取文件后缀名
slide_ext = os.path.splitext(slide_file_path)[1].lower()

if slide_ext == '.sdpc':
    # 注意这里：改成了首字母大写的 OpenSdpc
    wsi = opensdpc.OpenSdpc(slide_file_path) 
else:
    # 标准格式依然走 openslide
    wsi = openslide.open_slide(slide_file_path)
```

### 1.4 依赖安装

```bash
pip install opensdpc
```

---

## 二、Task 配置修改

### 2.1 背景

原始 CLAM 项目仅支持 `task_1_tumor_vs_normal`（肿瘤 vs 正常二分类）和 `task_2_tumor_subtyping`（肿瘤亚型三分类）两个任务。本项目新增了 `task_3_dlbcl_coo`，用于 DLBCL 淋巴瘤的 GCB vs non-GCB 分型任务。

### 2.2 修改的文件

| 文件路径 | 修改内容 |
|----------|----------|
| `main.py` | 新增 task 选项、Dataset 配置 |
| `eval.py` | 支持新 task 的评估 |
| `create_splits_seq.py` | 支持新 task 的数据划分 |

### 2.3 核心修改详解

#### 2.3.1 main.py - Task 参数定义

**位置**: `main.py` 第 130 行

```python
parser.add_argument('--task', type=str, 
    choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'task_3_dlbcl_coo'])
```

#### 2.3.2 main.py - Dataset 配置

**位置**: `main.py` 第 225-236 行

```python
elif args.task == 'task_3_dlbcl_coo':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/gcb_vs_nongcb.csv',
        data_dir=os.path.join(args.data_root_dir, 'dlbcl_resnet_features'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'GCB': 0, 'non-GCB': 1},
        patient_strat=False,
        ignore=[]
    )
```

**配置说明**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `csv_path` | `dataset_csv/gcb_vs_nongcb.csv` | DLBCL 分类数据集 CSV |
| `data_dir` | `dlbcl_resnet_features` | 特征文件目录 |
| `n_classes` | 2 | 二分类任务（GCB vs non-GCB）|
| `label_dict` | `{'GCB': 0, 'non-GCB': 1}` | 标签映射 |
| `patient_strat` | `False` | 按 slide 级别划分（非 patient 级别）|

### 2.4 数据集格式

#### 2.4.1 CSV 文件格式

**文件**: `dataset_csv/gcb_vs_nongcb.csv`

```csv
case_id,slide_id,label
1375314,1375314A01#3_1,GCB
1375314,1375314A02#3_2,GCB
...
1386008,1386008A01#3_9,non-GCB
...
```

| 列名 | 说明 |
|------|------|
| `case_id` | 患者 ID |
| `slide_id` | 切片 ID（含后缀 `#3_X`）|
| `label` | 标签：`GCB` 或 `non-GCB` |

#### 2.4.2 特征文件目录结构

```
features/
└── dlbcl_resnet_features/
    ├── pt_files/
    │   ├── 1375314A01#3_1.pt
    │   └── ...
    └── h5_files/
        ├── 1375314A01#3_1.h5
        └── ...
```

### 2.5 数据划分目录

```
splits/
└── task_3_dlbcl_coo_100/       # label_frac=1.0 → 100
    ├── splits_0.csv
    ├── splits_1.csv
    └── ...
    └── splits_9.csv            # 10-Fold 交叉验证
```

---

## 三、训练与评估命令

### 3.1 训练命令

```bash
# CLAM_SB (单分支) + weighted_sample
CUDA_VISIBLE_DEVICES=0 python main.py \
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

# CLAM_SB (无 weighted_sample)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --task task_3_dlbcl_coo \
  --data_root_dir /home/shanyiye/CLAM/features \
  --results_dir /home/shanyiye/CLAM/results \
  --exp_code dlbcl_gcb_nongcb_clam_sb_nows \
  --model_type clam_sb \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --bag_loss ce \
  --inst_loss svm \
  --log_data \
  --embed_dim 1024

# MIL (基线模型)
CUDA_VISIBLE_DEVICES=1 python main.py \
  --task task_3_dlbcl_coo \
  --data_root_dir /home/shanyiye/CLAM/features \
  --results_dir /home/shanyiye/CLAM/results \
  --exp_code dlbcl_gcb_nongcb_mil \
  --model_type mil \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --bag_loss ce \
  --log_data \
  --embed_dim 1024
```

### 3.2 评估命令

```bash
# 评估 CLAM_SB
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
```

---

## 四、注意事项

1. **opensdpc 库**: 使用 `.sdpc` 格式时必须安装 `opensdpc` 库
2. **特征提取**: 确保在特征提取时指定 `--slide_ext .sdpc` 参数
3. **CSV 格式**: 某些情况下 pandas 读取 CSV 会将 slide_id 转为数值型，需使用 `dtype=` 参数保持字符串格式
4. **patient_strat**: 当前 `task_3_dlbcl_coo` 使用 `patient_strat=False`，即按 slide 级别划分数据

---

*本文档最后更新于 2026-03-14*
