# 项目理解总览

> 最后更新：2026-04-15
> 目标：作为毕业论文项目的稳定认知底稿，后续可在此基础上持续修订。

## 1. 项目目标

### 1.1 当前项目的实际目标

基于 CLAM（Clustering-constrained Attention Multiple Instance Learning）框架，完成一个面向 DLBCL COO 分型的弱监督病理图像分类项目。当前仓库中的核心任务不是原始 CLAM README 里的示例任务，而是新增的 `task_3_dlbcl_coo`：

- 输入：WSI 提取后的 patch-level 特征
- 标签：slide/case 级别的 `GCB` vs `non-GCB`
- 方法：MIL / CLAM_SB / CLAM_MB
- 输出：分类结果、交叉验证结果、错误分析、热图可解释性结果

对应文件：

- 原始方法入口：[README.md](README.md)
- 中文项目综述：[project_summary.md](project_summary.md)
- DLBCL/SDPC 定制说明：[sdpc_modifications.md](sdpc_modifications.md)
- 训练主入口：[main.py](main.py)

### 1.2 与原始 CLAM 的关系

当前仓库是“原始 CLAM + DLBCL 论文项目定制”：

- 保留了 CLAM 的核心训练框架、模型结构和热图流程
- 新增了 `.sdpc` 格式 WSI 支持
- 新增了 `task_3_dlbcl_coo`
- 新增了多数据集选择：`nanchang` / `morph` / `tcga` / `all`
- 新增了面向论文实验稳定性的改进：
  - `val_auc` 监控
  - source-aware split
  - 特征增强
  - PCA 降维
  - warmup / attention entropy / label smoothing

当前项目应理解为：以 CLAM 为基础方法框架，围绕 DLBCL 分型论文目标做的实验系统和代码定制。

## 2. 目录与关键脚本说明

### 2.1 核心目录

| 目录 | 作用 | 说明 |
| --- | --- | --- |
| `dataset_csv/` | 数据标签表 | DLBCL 各数据集 CSV、合并表、说明文件 |
| `features/` | 提取后的特征 | 训练时主要直接读取 `pt_files/*.pt` |
| `splits/` | 交叉验证划分 | `task_3_dlbcl_coo_*_100` 为当前论文主任务 |
| `results/` | 训练输出 | 每个实验一个目录，含 checkpoint、split 副本、summary |
| `eval_results/` | 独立评估输出 | 每折预测结果与汇总表 |
| `analysis_results/` | 进一步分析结果 | OOF 指标、错例、ROC、混淆矩阵 |
| `heatmaps/` | 可解释性可视化 | 热图配置、原始热图、中间结果、导出结果 |
| `models/` | 模型定义 | CLAM_SB / CLAM_MB / MIL |
| `dataset_modules/` | 数据集定义 | patient-level split、特征加载、增强入口 |
| `utils/` | 训练与工具函数 | early stopping、split 生成、PCA、feature aug |
| `wsi_core/` | WSI 处理 | patch 提取、组织分割、热图生成 |

### 2.2 关键脚本

| 脚本 | 作用 | 当前论文中的地位 |
| --- | --- | --- |
| [main.py](main.py) | 训练入口 | 论文核心实验入口 |
| [eval.py](eval.py) | 独立评估 | 生成 `eval_results` |
| [create_splits_seq.py](create_splits_seq.py) | 生成交叉验证划分 | 控制 patient-level/source-aware 设计 |
| [extract_features_fp.py](extract_features_fp.py) | 快速特征提取 | 使用 UNI / ResNet 等特征 |
| [create_patches_fp.py](create_patches_fp.py) | patch 坐标提取 | WSI 预处理起点 |
| [create_heatmaps.py](create_heatmaps.py) | 热图生成 | 论文解释性部分候选 |
| [analyze_eval_results.py](analyze_eval_results.py) | 结果分析 | 生成错误样本和 OOF 指标 |

### 2.3 关键代码改动点

| 文件 | 关键改动 |
| --- | --- |
| [main.py](main.py) | 新增 `task_3_dlbcl_coo`、多数据集参数、训练策略参数 |
| [eval.py](eval.py) | 支持 DLBCL 数据集、`ckpt_type`、`auto_fix_inversion` |
| [create_splits_seq.py](create_splits_seq.py) | DLBCL 多数据集 split，`all` 支持 source-aware |
| [dataset_modules/dataset_generic.py](dataset_modules/dataset_generic.py) | patient/source 聚合，增强与 PCA 入口 |
| [utils/core_utils.py](utils/core_utils.py) | `val_auc` 监控、双 checkpoint 跟踪、训练策略增强 |
| [utils/utils.py](utils/utils.py) | source-aware split 实现、loader 增强入口 |
| [utils/pca_utils.py](utils/pca_utils.py) | 训练集拟合 PCA |
| [utils/feature_aug.py](utils/feature_aug.py) | 高斯噪声、feature dropout、patch dropout |
| [wsi_core/WholeSlideImage.py](wsi_core/WholeSlideImage.py) | `.sdpc` 支持 |
| [extract_features_fp.py](extract_features_fp.py) | `.sdpc` 特征提取支持 |

## 3. 数据与任务定义

### 3.1 主任务定义

当前论文主任务为：

- 任务名：`task_3_dlbcl_coo`
- 分类目标：`GCB` vs `non-GCB`
- 学习方式：弱监督 MIL / CLAM
- 划分方式：以 patient 为单位划分 train/val/test，避免同一患者跨集合泄露

实现入口：

- [main.py](main.py)
- [eval.py](eval.py)
- [create_splits_seq.py](create_splits_seq.py)

### 3.2 数据集清单

| 数据集 | CSV | slides | cases | 备注 |
| --- | --- | ---: | ---: | --- |
| nanchang | [dataset_csv/nanchang_dlbcl.csv](dataset_csv/nanchang_dlbcl.csv) | 389 | 50 | 单中心数据，病例少，slide 多 |
| morph | [dataset_csv/dlbcl_morph.csv](dataset_csv/dlbcl_morph.csv) | 183 | 132 | 标签来自形态/Hans 推导链路 |
| tcga | [dataset_csv/tcga_dlbcl.csv](dataset_csv/tcga_dlbcl.csv) | 39 | 39 | 外部数据量小 |
| all | [dataset_csv/dlbcl_all.csv](dataset_csv/dlbcl_all.csv) | 611 | 221 | 合并三源，并带 `source` 列 |

补充说明：

- `all` 数据集说明见 [dataset_csv/dlbcl_all_description](dataset_csv/dlbcl_all_description)
- `all` 的特征目录 [features/all_uni_features](features/all_uni_features) 采用软链接整合多源特征，不是重新单独提取一套物理文件

### 3.3 特征定义

训练时主要使用预提取特征，而非原始 patch 图像：

- 数据格式：`pt_files/<slide_id>.pt`
- 特征维度：
  - ResNet 截断特征通常为 1024
  - UNI 特征为 1024
- 当前论文主线明显偏向 UNI 特征

现有特征目录：

- [features/nanchang_uni_features](features/nanchang_uni_features)
- [features/morph_uni_features](features/morph_uni_features)
- [features/tcga_uni_features](features/tcga_uni_features)
- [features/all_uni_features](features/all_uni_features)

### 3.4 标签层级

当前训练数据在 CSV 中按 slide 记录，但 split 使用 patient-level：

- CSV 中至少包含 `case_id`, `slide_id`, `label`
- `all` 还包含 `source`
- 训练时读入的是 slide-level bag 特征
- 划分时先聚合 patient，再回写到对应 slide

这对论文非常重要，因为它决定了实验是否存在数据泄露。

## 4. 实验流水线

### 4.1 完整流程

1. 原始 WSI 组织分割和 patch 坐标提取  
   对应脚本：[create_patches_fp.py](create_patches_fp.py)

2. patch 特征提取  
   对应脚本：[extract_features_fp.py](extract_features_fp.py)

3. 生成 patient-level 交叉验证划分  
   对应脚本：[create_splits_seq.py](create_splits_seq.py)

4. 训练 MIL / CLAM 模型  
   对应脚本：[main.py](main.py)

5. 评估每折结果  
   对应脚本：[eval.py](eval.py)

6. 聚合分析错误案例、ROC、混淆矩阵  
   对应目录：[analysis_results](analysis_results)

7. 生成注意力热图和高注意力 patch  
   对应脚本：[create_heatmaps.py](create_heatmaps.py)

### 4.2 当前代码中的训练增强链路

训练阶段除原始 CLAM 外，已经加入以下增强或约束：

- `monitor_metric=val_auc`
- feature noise
- feature dropout
- patch dropout
- max patches per bag
- PCA 降维
- warmup bag-only epochs
- attention entropy regularization
- label smoothing
- source-aware split

这说明论文目前的重点已从“能否跑通 CLAM”转向“如何让实验更稳定、更可信”。

### 4.3 可解释性链路

现有热图配置与结果主要集中在早期 DLBCL 基线：

- 配置目录：[heatmaps/configs](heatmaps/configs)
- 已有配置：
  - [heatmaps/configs/dlbcl_clam_sb.yaml](heatmaps/configs/dlbcl_clam_sb.yaml)
  - [heatmaps/configs/dlbcl_clam_mb.yaml](heatmaps/configs/dlbcl_clam_mb.yaml)
  - 以及 nows / mil 版本
- 结果目录：
  - [heatmaps/heatmap_production_results](heatmaps/heatmap_production_results)
  - [heatmaps/heatmap_raw_results](heatmaps/heatmap_raw_results)

目前仓库内没有看到针对最新 `morph_uni_clam_sb_v3` 的热图产物。

## 5. 当前进展

### 5.1 已经完成的阶段

已确认完成：

- 原始 CLAM 到 DLBCL 任务的代码迁移
- `.sdpc` 支持
- 多数据集 DLBCL 训练和评估接口
- UNI 特征提取与训练链路
- patient-level split
- `all` 数据集 source-aware split
- 训练流程正则化增强
- 若干轮基线实验和优化实验

### 5.2 可确认的关键结果

#### 早期 UNI 基线

| 实验 | 目录 | mean test AUC | 说明 |
| --- | --- | ---: | --- |
| nanchang UNI + CLAM_SB | [results/nanchang_uni_clam_sb_s1](results/nanchang_uni_clam_sb_s1) | 0.6599 | 单中心基线 |
| morph UNI + CLAM_SB | [results/morph_uni_clam_sb_s1](results/morph_uni_clam_sb_s1) | 0.6568 | 单域基线 |
| all UNI + CLAM_SB | [results/dlbcl_all_uni_clam_sb_s1](results/dlbcl_all_uni_clam_sb_s1) | 0.5367 | 多源混合后明显变难 |

#### 早期 DLBCL ResNet 基线

| 实验 | 目录 | mean test AUC | 说明 |
| --- | --- | ---: | --- |
| CLAM_SB noWS | [results/dlbcl_gcb_nongcb_clam_sb_nows_s1](results/dlbcl_gcb_nongcb_clam_sb_nows_s1) | 约 0.6056 | 旧版 DLBCL baseline |
| CLAM_SB WS | [results/dlbcl_gcb_nongcb_clam_sb_s1](results/dlbcl_gcb_nongcb_clam_sb_s1) | 约 0.5568 | 旧版 DLBCL baseline |
| CLAM_MB | [results/dlbcl_gcb_nongcb_clam_mb_s1](results/dlbcl_gcb_nongcb_clam_mb_s1) | 约 0.5817 | 旧版 DLBCL baseline |
| MIL | [results/dlbcl_gcb_nongcb_mil_s1](results/dlbcl_gcb_nongcb_mil_s1) | 约 0.4994 | 基线较弱 |

这些数值与 [analysis_results/model_comparison.csv](analysis_results/model_comparison.csv) 和各 `report.txt` 基本一致。

#### 后期优化实验

| 实验 | 目录 | mean test AUC | 状态 |
| --- | --- | ---: | --- |
| morph UNI CLAM_SB v2 | [results/morph_uni_clam_sb_v2_s1](results/morph_uni_clam_sb_v2_s1) | 0.7461 | 已完成 |
| morph UNI CLAM_SB strong v1 | [results/morph_uni_clam_sb_strong_v1_s1](results/morph_uni_clam_sb_strong_v1_s1) | 0.7655 | 已完成 |
| morph UNI CLAM_SB v3 | [results/morph_uni_clam_sb_v3_s1](results/morph_uni_clam_sb_v3_s1) | 0.7806 | 当前仓库内最佳完整结果 |
| nanchang UNI CLAM_SB v2 | [results/nanchang_uni_clam_sb_v2_s1](results/nanchang_uni_clam_sb_v2_s1) | 0.4269 | 已完成，但效果不理想 |
| nanchang UNI CLAM_SB reg | [results/nanchang_uni_clam_sb_reg_s1](results/nanchang_uni_clam_sb_reg_s1) | 0.6144 | 已完成，且有 fixed eval |
| all UNI CLAM_SB v2 | [results/all_uni_clam_sb_v2_s1](results/all_uni_clam_sb_v2_s1) | 待汇总 | 目录不完整，缺最终 `summary.csv` |

### 5.3 当前最像论文主结果的方向

基于仓库现有结果，当前最适合发展为论文主线的是：

- 主实验数据集：`morph`
- 主模型：UNI + CLAM_SB
- 主结果：`morph_uni_clam_sb_v3`

原因：

- 结果完整
- 相比早期基线提升明显
- 训练参数和改进链路已经沉淀到代码中
- 已有消融实验规划文档与之配套

这一判断来自仓库现有材料，不等于导师最终定稿意见。

## 6. 与论文各章节的映射

### 第 1 章 绪论

可写内容：

- DLBCL COO 分型的临床意义
- WSI 级病理分析的价值
- patch 标注成本高，弱监督是现实需求
- 以 CLAM 为代表的弱监督 MIL 方法适合该任务

当前仓库支撑程度：

- 方法背景足够
- 临床背景材料不足，需补外部文献

### 第 2 章 相关工作

可写内容：

- MIL / attention MIL / CLAM
- 计算病理中的 foundation model 特征（如 UNI）
- 病理弱监督学习和可解释性方法

当前仓库支撑程度：

- CLAM 和工程流程证据充足
- 文献综述材料仍需单独补齐

### 第 3 章 数据与方法

可直接对应仓库内容：

- 数据集定义与来源
- patient-level split 设计
- 特征提取流程
- CLAM / MIL 模型结构
- DLBCL 定制改动
- source-aware split
- 训练策略优化

主要依据：

- [main.py](main.py)
- [create_splits_seq.py](create_splits_seq.py)
- [dataset_modules/dataset_generic.py](dataset_modules/dataset_generic.py)
- [utils/core_utils.py](utils/core_utils.py)
- [utils/feature_aug.py](utils/feature_aug.py)
- [utils/pca_utils.py](utils/pca_utils.py)

### 第 4 章 实验设计与结果

可直接组织为：

- 数据集与评估协议
- 基线实验：MIL / CLAM，ResNet / UNI
- 单域结果：nanchang, morph
- 混合域结果：all
- 训练改进结果：v2 / v3
- 错误分析与稳定性分析

现有证据：

- `results/`
- `eval_results/`
- `analysis_results/`

### 第 5 章 可解释性与讨论

可写内容：

- 热图展示
- 高注意力 patch 分析
- 聚类反转问题
- source shift / small-sample instability
- morph 标签性质对结论边界的影响

对应材料：

- [docs/clustering_inversion_fix.md](docs/clustering_inversion_fix.md)
- [heatmaps](heatmaps)
- [analysis_results](analysis_results)

### 第 6 章 总结与展望

当前可预期结论：

- CLAM 在 DLBCL COO 弱监督任务上能学到一定信号
- 实验稳定性严重依赖 split 设计、选模指标和正则化策略
- 单域 `morph` 方向目前优于多源混合 `all`
- 论文主结论应谨慎区分“可用信号”与“跨域泛化能力”

## 7. 当前缺口

### 7.1 论文材料层面的缺口

- 仓库中暂未看到毕业论文任务书、开题报告、中期检查材料、章节草稿
- 论文最终主线尚未由现有材料明确锁定
- 临床背景、相关工作、参考文献仍需后续补充

### 7.2 实验层面的缺口

- `all_uni_clam_sb_v2_s1` 尚未完整收口
- 消融实验大多仍是计划，未形成完整表格结果
- 最新 `v2/v3` 结果尚未看到系统性的 `eval_results` 与 `analysis_results`
- 最新最佳模型 `morph_uni_clam_sb_v3` 尚未看到对应热图

### 7.3 证据链层面的缺口

- [docs/训练效果问题分析与改进建议.md](docs/训练效果问题分析与改进建议.md) 中引用了 `analysis_results/tfevents_fold_summary.csv`，当前仓库未见该文件
- 原始 WSI 目录 [raw_slides](raw_slides) 当前基本为空；若后续需要重跑切片或热图，需确认原始数据是否在仓库外部
- 旧版 `eval_results` 多数对应 `s1` 基线，未覆盖最新结果

### 7.4 研究结论边界的缺口

- `morph` 标签并非直接分子金标准，这会影响论文措辞
- `all` 的跨源结论尚不稳定
- `nanchang` 病例数少，fold 波动大，不适合直接作为唯一主结论

## 8. 下一步建议

### 8.1 论文主线建议

建议优先采用“双层主线”：

1. 主结论线：以 `morph_uni_clam_sb_v3` 为当前最佳完整结果，形成方法改进与效果提升的主叙事
2. 支撑结论线：用 `nanchang` 与 `all` 说明小样本和跨域混合下的稳定性问题与局限

### 8.2 近期优先任务

优先级建议：

1. 先明确论文主实验是否以 `morph` 为主
2. 为 `morph_uni_clam_sb_v3` 补一套正式评估与分析结果
3. 补关键消融实验：
   - MIL baseline
   - no PCA
   - no aug
   - bag-only / no instance clustering
4. 视时间决定是否补完 `all_uni_clam_sb_v2`
5. 为最终主模型补热图和典型病例图

### 8.3 文档管理建议

后续建议把实验按三类管理：

- 已完成且可写入论文
- 已跑过但仅作探索参考
- 尚未完成/待复现

对应工作台文档见后续新增的 [ongoing_thesis_workspace.md](ongoing_thesis_workspace.md)。

## 9. 当前结论

基于仓库现有真实材料，可以形成的稳妥结论是：

- 本项目已经从“CLAM 代码复现”进入“围绕 DLBCL 毕业论文目标的实验系统建设阶段”
- 最成熟的论文结果方向不是早期 DLBCL ResNet 基线，也不是 `all` 混合域实验，而是 `morph` 上的 UNI + CLAM_SB 优化线
- 当前最大问题不是代码跑不通，而是：
  - 哪条结果线作为论文主结论
  - 哪些实验还需要补齐证据链
  - 哪些结论必须在论文中谨慎表述
