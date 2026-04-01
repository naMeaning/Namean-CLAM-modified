# CLAM 项目总结文档

> **CLAM**: Clustering-constrained Attention Multiple Instance Learning  
> **论文**: *Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images* (Lu et al., Nature Biomedical Engineering, 2021)

---

## 一、项目总结

### 1.1 背景与问题定位

CLAM 解决的核心问题：**在仅有 slide 级别标签（弱监督）的条件下，对全切片病理图像（WSI）进行分类**。

| 传统 MIL 的局限 | CLAM 的改进 |
|---|---|
| 仅使用 bag-level 损失 | 同时引入 instance-level 聚类损失 |
| 注意力机制无约束 | 通过 top-k 正负样本强化注意力集中性 |
| 单一关注分支 | 支持多分支（每类独立注意力）|

### 1.2 整体数据流程

```
WSI 文件 (.svs / .sdpc / .tiff)
    │
    ▼ [create_patches.py / create_patches_fp.py]
组织切割（segmentTissue）→ Patch 坐标提取（process_contour）
    │ 输出：h5_files/<slide_id>.h5 （坐标）
    │
    ▼ [extract_features.py]
预训练编码器推理（ResNet-50 / UNI / CONCH）
    │ 输出：h5_files/<slide_id>.h5 (特征+坐标)
    │       pt_files/<slide_id>.pt  (特征，MIL 训练用)
    │
    ▼ [create_splits_seq.py]
K-Fold 数据划分
    │ 输出：splits/<task>_<label_frac>/ 目录下的 splits_0.csv ... splits_9.csv
    │
    ▼ [main.py]
CLAM / MIL 训练（K-Fold 交叉验证）
    │ 输出：results/<exp_code>_s<seed>/ 下的 checkpoint、pkl、summary.csv
    │
    ▼ [eval.py] (可选)
单次独立评估
    │
    ▼ [create_heatmaps.py]
注意力热图可视化
```

### 1.3 各模块功能对应表

| 文件 | 功能 | 核心类/函数 |
|---|---|---|
| `main.py` | 训练主入口，K-Fold 循环 | `main()`, `seed_torch()` |
| `models/model_clam.py` | CLAM 单/多分支模型 | `CLAM_SB`, `CLAM_MB`, `Attn_Net_Gated` |
| `models/model_mil.py` | 基础 MIL 模型 | `MIL_fc`, `MIL_fc_mc` |
| `utils/core_utils.py` | 训练/验证/测试循环 | `train()`, `EarlyStopping`, `Accuracy_Logger` |
| `utils/utils.py` | 通用工具 | `generate_split()`, `get_split_loader()`, `get_optim()` |
| `utils/eval_utils.py` | 独立评估 | `initiate_model()`, `eval()`, `summary()` |
| `dataset_modules/dataset_generic.py` | 数据集类 | `Generic_MIL_Dataset`, `Generic_Split` |
| `wsi_core/WholeSlideImage.py` | WSI 处理 | `segmentTissue()`, `process_contour()`, `visHeatmap()` |
| `extract_features.py` | 特征提取 | `compute_w_loader()` |

### 1.4 模型输入输出格式

| 阶段 | 输入 | 输出 |
|---|---|---|
| 特征提取 | `(N, 3, 224, 224)` patch 图像批次 | `(N, D)` 特征矩阵，D=1024 |
| CLAM 前向 | `(N, D)` bag 特征矩阵（N=patch 数） | `logits`, `Y_prob`, `Y_hat`, `A_raw`, `results_dict` |
| 实例评估 | attention `A`, 特征 `h`, top-k 采样 | `instance_loss`, `inst_preds`, `inst_labels` |

---

## 二、复现建议

### 2.1 环境依赖安装

```bash
# 创建 conda 环境
conda env create -f env.yml
conda activate clam

# 若使用 .sdpc 格式 WSI，需额外安装 opensdpc：
pip install opensdpc

# （可选）若使用 UNI / CONCH 编码器，按 README 申请权限并安装
```

### 2.2 数据准备

**CSV 文件格式**（`dataset_csv/` 目录）：

| 列名 | 说明 |
|---|---|
| `case_id` | 患者 ID（用于 patient-level 划分） |
| `slide_id` | 切片 ID（文件名，不含扩展名） |
| `label` | 字符串标签（如 `tumor_tissue`, `normal_tissue`） |

**目录结构**：
```
data_root_dir/
├── tumor_vs_normal_resnet_features/
│   ├── pt_files/    # <slide_id>.pt
│   └── h5_files/    # <slide_id>.h5
```

### 2.3 完整运行命令示例

```bash
# 步骤 1：Patch 切割（fp 版本支持多进程）
python create_patches_fp.py \
  --source raw_slides/ \
  --save_dir results/ \
  --patch_size 256 --seg --patch 

# 步骤 2：特征提取（使用 ResNet-50 截断特征）
python extract_features_fp.py \
  --data_dir RESULTS_DIRECTORY \
  --csv_path dataset_csv/tumor_vs_normal_dummy_clean.csv \
  --feat_dir FEATURES_DIRECTORY \
  --model_name resnet50_trunc \
  --batch_size 512

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
  --data_h5_dir results \
  --data_slide_dir raw_slides \
  --csv_path results/csv_part1.csv \
  --feat_dir features \
  --batch_size 512 \
  --slide_ext .sdpc

  CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py \
  --data_h5_dir results \
  --data_slide_dir raw_slides \
  --csv_path results/csv_part2.csv \
  --feat_dir features \
  --batch_size 512 \
  --slide_ext .sdpc
# 步骤 3：生成 K-Fold 数据划分
python create_splits_seq.py \
  --task task_1_tumor_vs_normal \
  --seed 1 --label_frac 1 \
  --k 10

# 步骤 4：训练（CLAM 单分支，CE 损失）
python main.py \
  --drop_out 0.25 --early_stopping \
  --lr 2e-4 --k 10 \
  --exp_code task1_tumor_vs_normal_CLAM_SB \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb --model_size small \
  --bag_loss ce \
  --data_root_dir FEATURES_DIRECTORY \
  --inst_loss svm --B 8


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


# 步骤 5：独立评估
python eval.py \
  --drop_out 0.25 --k 10 \
  --exp_code task1_tumor_vs_normal_CLAM_SB \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb \
  --results_dir results/ \
  --data_root_dir FEATURES_DIRECTORY


1. CLAM_SB + weighted_sample
对应目录：
/home/shanyiye/CLAM/results/dlbcl_gcb_nongcb_clam_sb_s1

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
2. CLAM_SB + no weighted_sample
对应目录：
/home/shanyiye/CLAM/results/dlbcl_gcb_nongcb_clam_sb_nows_s1

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --k 10 \
  --models_exp_code dlbcl_gcb_nongcb_clam_sb_nows_s1 \
  --save_exp_code dlbcl_gcb_nongcb_clam_sb_nows_eval \
  --task task_3_dlbcl_coo \
  --model_type clam_sb \
  --results_dir /home/shanyiye/CLAM/results \
  --data_root_dir /home/shanyiye/CLAM/features \
  --embed_dim 1024 \
  --seed 1
3. MIL baseline
对应目录：
/home/shanyiye/CLAM/results/dlbcl_gcb_nongcb_mil_s1

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --k 10 \
  --models_exp_code dlbcl_gcb_nongcb_mil_s1 \
  --save_exp_code dlbcl_gcb_nongcb_mil_eval \
  --task task_3_dlbcl_coo \
  --model_type mil \
  --results_dir /home/shanyiye/CLAM/results \
  --data_root_dir /home/shanyiye/CLAM/features \
  --embed_dim 1024 \
  --seed 1


这 3 条跑完后，结果一般会在：

/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_eval
/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_nows_eval
/home/shanyiye/CLAM/eval_results/EVAL_dlbcl_gcb_nongcb_mil_eval

```




### 2.4 常见问题排查

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `slide_id` 匹配失败 | CSV 中数字 ID 被 `read_csv` 转为数值型 | 已在代码中通过 `dtype=` 参数修复，确保 CSV 格式正确 |
| `instance_loss_fn` 加载报错 | checkpoint 含损失函数 key | `eval_utils.initiate_model()` 已自动过滤 |
| GPU 内存不足 | batch_size 过大 | 降低 `--batch_size`，或使用 `--model_size small` |
| `.sdpc` 加载失败 | 未安装 opensdpc | `pip install opensdpc` |
| 早停不触发 | `stop_epoch=50` 保护期 | 调整 `EarlyStopping(stop_epoch=...)` 参数 |

---

## 三、科研改进与创新点建议

### 3.1 模型层面

| 方向 | 具体建议 | 预期收益 |
|---|---|---|
| **更强注意力机制** | 将 `Attn_Net_Gated` 替换为 Transformer Self-Attention（如 TransMIL、ABMIL+Transformer） | 建模 patch 间长程空间依赖 |
| **多尺度特征融合** | 在不同金字塔层级（20x/40x）分别提取特征，再融合 | 兼顾细胞级和组织级上下文 |
| **图神经网络** | 将 patch 构建为空间近邻图（GNN-MIL） | 显式利用 patch 空间拓扑关系 |
| **原型网络** | 用可学习的类原型替换 instance 聚类正负样本采样 | 更稳定的实例级监督 |
| **不确定性估计** | 加入 MC Dropout / Evidential Deep Learning | 输出预测置信度，提高临床可信度 |

### 3.2 数据层面

| 方向 | 具体建议 | 预期收益 |
|---|---|---|
| **更强预训练特征** | 使用病理专用大模型（UNI、CONCH、PLIP、CTransPath）替换 ResNet-50 | 特征判别力大幅提升，少样本性能强 |
| **多模态融合** | 融合基因组/临床信息（SurvPath、PORPOISE 框架） | 多模态互补，预后/分型更准 |
| **数据增强** | 在 bag 级别进行随机 patch 丢弃/重复（StochMIL） | 缓解过拟合，提高泛化性 |
| **半监督/自监督** | 用无标注 WSI 进行对比学习预训练 | 降低标注成本 |
| **主动学习** | 基于注意力分数选择最具信息量的 slide 由病理医生标注 | 标注效率最大化 |

### 3.3 工程层面

| 方向 | 具体建议 | 预期收益 |
|---|---|---|
| **推理加速** | 将模型导出为 ONNX / TorchScript，使用 TensorRT 加速 | 临床部署延迟降低 |
| **流式特征提取** | 边切 patch 边推理（在线特征提取），无需预存所有特征 | 大幅减少磁盘占用 |
| **分布式训练** | 使用 DDP（DistributedDataParallel）支持多卡训练 | 加快大规模实验 |
| **可解释性增强** | 将注意力热图叠加到报告生成管线，输出结构化病理报告 | 提升临床可用性 |
| **端到端训练** | 联合训练特征提取器和 MIL 模型（内存高效梯度累积） | 避免特征提取与分类的优化目标不一致 |
