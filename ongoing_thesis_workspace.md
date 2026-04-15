# 论文持续工作台

> 最后更新：2026-04-15
> 用途：作为毕业论文长期协作工作台，持续更新事实、待办、风险和实验索引。

## 1. 总目标

围绕 CLAM/DLBCL 项目，完成一篇可提交的毕业论文，并保证：

- 结论基于真实代码、真实实验和真实结果
- 论文叙事与仓库中实际完成的工作一致
- 所有关键结果都能追溯到具体目录和文件
- 不确定项明确标注，不用想当然补空白

当前建议的总体路线：

- 以 DLBCL COO 二分类为论文中心任务
- 以 CLAM 为基础方法框架
- 以 `morph` 方向的优化结果作为当前最成熟主线
- 以 `nanchang` 和 `all` 结果支撑对稳定性、泛化和局限性的讨论

## 2. 当前进度

### 2.1 已完成

- 已完成对仓库核心材料的第一轮系统阅读
- 已确认当前论文任务不是泛泛的 CLAM 复现，而是 DLBCL 定制项目
- 已确认数据集、主任务、主要脚本、实验产物目录
- 已确认一条相对完整的优化实验主线：
  - `morph_uni_clam_sb_s1`
  - `morph_uni_clam_sb_v2_s1`
  - `morph_uni_clam_sb_strong_v1_s1`
  - `morph_uni_clam_sb_v3_s1`
- 已发现当前仓库中存在实验完整度不一致的问题

### 2.2 进行中

- 建立论文工作台和项目理解文档
- 梳理“哪些结果可直接写论文，哪些只能当探索性结果”

### 2.3 尚未完成

- 论文任务书/开题/中期材料整合
- 论文章节初稿整合
- 最新最佳模型的完整评估、可视化和图表整理
- 消融实验补齐

## 3. 已确认事实

### 3.1 项目与代码

- 项目基于 CLAM 改造，核心训练入口是 [main.py](main.py)
- 主任务为 `task_3_dlbcl_coo`
- 已支持 `.sdpc` 格式 WSI
- 数据划分采用 patient-level，`all` 数据集支持 source-aware split
- 当前训练流程已加入多个防过拟合与稳定性策略

### 3.2 数据

- `nanchang`: 389 slides / 50 cases
- `morph`: 183 slides / 132 cases
- `tcga`: 39 slides / 39 cases
- `all`: 611 slides / 221 cases

关键数据文件：

- [dataset_csv/nanchang_dlbcl.csv](dataset_csv/nanchang_dlbcl.csv)
- [dataset_csv/dlbcl_morph.csv](dataset_csv/dlbcl_morph.csv)
- [dataset_csv/tcga_dlbcl.csv](dataset_csv/tcga_dlbcl.csv)
- [dataset_csv/dlbcl_all.csv](dataset_csv/dlbcl_all.csv)

### 3.3 特征

- 当前主训练材料是特征文件，不是原始 patch 图像
- 已确认存在 UNI 特征目录：
  - [features/nanchang_uni_features](features/nanchang_uni_features)
  - [features/morph_uni_features](features/morph_uni_features)
  - [features/tcga_uni_features](features/tcga_uni_features)
  - [features/all_uni_features](features/all_uni_features)
- `all_uni_features/pt_files` 采用软链接整合多源特征

### 3.4 结果

- 早期单域基线：
  - `nanchang_uni_clam_sb_s1` mean test AUC = 0.6599
  - `morph_uni_clam_sb_s1` mean test AUC = 0.6568
- 多源基线：
  - `dlbcl_all_uni_clam_sb_s1` mean test AUC = 0.5367
- 目前最佳完整结果：
  - `morph_uni_clam_sb_v3_s1` mean test AUC = 0.7806

### 3.5 已确认但需谨慎表述的事实

- `morph` 方向目前效果最好，但其标签链路并非一定等价于分子金标准
- `nanchang` 的病例数少，fold 波动大
- `all` 混合域结果比单域差，说明跨源偏移问题显著

## 4. 待确认问题

以下问题目前无法仅凭仓库现有材料完全确认：

1. 论文最终主线是否确定为 `morph` 方向
2. 导师是否接受把 `morph` 作为主实验数据集
3. `morph` 标签的严格医学定义和使用边界
4. 论文任务书、开题报告、中期检查材料在哪里
5. 是否已有论文草稿
6. 原始 WSI 是否在仓库外部另存
7. `all_uni_clam_sb_v2_s1` 是否只是中断实验，还是另有完整结果在别处
8. 文档中提到的 `analysis_results/tfevents_fold_summary.csv` 是否在别处存在
9. 最终论文是否需要热图作为必备图
10. 最新模型是否还需要重新统一评估一次

## 5. 待办事项

### P0：论文主线确认

- [ ] 确认论文任务书/开题材料
- [ ] 确认论文最终主实验数据集和主结果线
- [ ] 确认哪些实验允许正式写入论文

### P0：结果证据链补齐

- [ ] 为 `morph_uni_clam_sb_v3_s1` 生成正式评估目录
- [ ] 为 `morph_uni_clam_sb_v3_s1` 生成错误分析和图表
- [ ] 确认 `v3` 是否需要重新跑 `eval.py`

### P1：关键消融实验

- [ ] MIL baseline
- [ ] CLAM bag-only / 关闭 instance clustering
- [ ] no PCA
- [ ] no augmentation
- [ ] bag_weight 对比

参考文档：[docs/ablation_experiments.md](docs/ablation_experiments.md)

### P1：论文图表准备

- [ ] 主结果表
- [ ] 消融实验表
- [ ] ROC 图 / confusion matrix
- [ ] 典型热图与高注意力 patch 图
- [ ] 训练稳定性或实验设计示意图

### P2：补充性工作

- [ ] 如有必要，补完 `all_uni_clam_sb_v2`
- [ ] 如有必要，重新组织 `nanchang` 的 5-fold 结果叙事
- [ ] 梳理旧版 `s1` 结果与新版 `v2/v3` 结果的边界

## 6. 实验记录索引

### 6.1 早期 DLBCL ResNet 基线

| 实验 | 目录 | 状态 | 备注 |
| --- | --- | --- | --- |
| CLAM_SB_WS | [results/dlbcl_gcb_nongcb_clam_sb_s1](results/dlbcl_gcb_nongcb_clam_sb_s1) | 已完成 | 旧版 baseline |
| CLAM_SB_noWS | [results/dlbcl_gcb_nongcb_clam_sb_nows_s1](results/dlbcl_gcb_nongcb_clam_sb_nows_s1) | 已完成 | 旧版 baseline |
| CLAM_MB_WS | [results/dlbcl_gcb_nongcb_clam_mb_s1](results/dlbcl_gcb_nongcb_clam_mb_s1) | 已完成 | 旧版 baseline |
| CLAM_MB_noWS | [results/dlbcl_gcb_nongcb_clam_mb_nows_s1](results/dlbcl_gcb_nongcb_clam_mb_nows_s1) | 已完成 | 旧版 baseline |
| MIL | [results/dlbcl_gcb_nongcb_mil_s1](results/dlbcl_gcb_nongcb_mil_s1) | 已完成 | 旧版 baseline |

对应评估与分析：

- [eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_eval](eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_eval)
- [eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_nows_eval](eval_results/EVAL_dlbcl_gcb_nongcb_clam_sb_nows_eval)
- [eval_results/EVAL_dlbcl_gcb_nongcb_clam_mb_eval](eval_results/EVAL_dlbcl_gcb_nongcb_clam_mb_eval)
- [eval_results/EVAL_dlbcl_gcb_nongcb_clam_mb_nows_eval](eval_results/EVAL_dlbcl_gcb_nongcb_clam_mb_nows_eval)
- [eval_results/EVAL_dlbcl_gcb_nongcb_mil_eval](eval_results/EVAL_dlbcl_gcb_nongcb_mil_eval)
- [analysis_results/model_comparison.csv](analysis_results/model_comparison.csv)

### 6.2 UNI 单域基线

| 实验 | 目录 | 状态 | 备注 |
| --- | --- | --- | --- |
| nanchang_uni_clam_sb_s1 | [results/nanchang_uni_clam_sb_s1](results/nanchang_uni_clam_sb_s1) | 已完成 | 10-fold 基线 |
| nanchang_uni_mil_s1 | [results/nanchang_uni_mil_s1](results/nanchang_uni_mil_s1) | 已完成 | MIL 对照 |
| morph_uni_clam_sb_s1 | [results/morph_uni_clam_sb_s1](results/morph_uni_clam_sb_s1) | 已完成 | 10-fold 基线 |
| dlbcl_all_uni_clam_sb_s1 | [results/dlbcl_all_uni_clam_sb_s1](results/dlbcl_all_uni_clam_sb_s1) | 已完成 | 合并域基线 |

对应评估：

- [eval_results/EVAL_nanchang_uni_clam_sb_eval](eval_results/EVAL_nanchang_uni_clam_sb_eval)
- [eval_results/EVAL_morph_uni_clam_sb_eval](eval_results/EVAL_morph_uni_clam_sb_eval)
- [eval_results/EVAL_dlbcl_all_uni_clam_sb_eval](eval_results/EVAL_dlbcl_all_uni_clam_sb_eval)

### 6.3 Nanchang 优化实验

| 实验 | 目录 | 状态 | 备注 |
| --- | --- | --- | --- |
| nanchang_uni_clam_sb_reg_s1 | [results/nanchang_uni_clam_sb_reg_s1](results/nanchang_uni_clam_sb_reg_s1) | 已完成 | 有 fixed eval |
| nanchang_uni_clam_sb_v2_s1 | [results/nanchang_uni_clam_sb_v2_s1](results/nanchang_uni_clam_sb_v2_s1) | 已完成 | 5-fold，但效果不理想 |

对应评估：

- [eval_results/EVAL_nanchang_eval_fixed](eval_results/EVAL_nanchang_eval_fixed)

### 6.4 Morph 优化实验

| 实验 | 目录 | 状态 | 备注 |
| --- | --- | --- | --- |
| morph_uni_clam_sb_v2_s1 | [results/morph_uni_clam_sb_v2_s1](results/morph_uni_clam_sb_v2_s1) | 已完成 | 明显优于早期基线 |
| morph_uni_clam_sb_strong_v1_s1 | [results/morph_uni_clam_sb_strong_v1_s1](results/morph_uni_clam_sb_strong_v1_s1) | 已完成 | 强正则版本 |
| morph_uni_clam_sb_v3_s1 | [results/morph_uni_clam_sb_v3_s1](results/morph_uni_clam_sb_v3_s1) | 已完成 | 当前最佳完整结果 |

### 6.5 All 优化实验

| 实验 | 目录 | 状态 | 备注 |
| --- | --- | --- | --- |
| all_uni_clam_sb_v2_s1 | [results/all_uni_clam_sb_v2_s1](results/all_uni_clam_sb_v2_s1) | 未完整收口 | 缺最终 `summary.csv`，已有 0-8 折部分产物 |

### 6.6 热图与解释性

| 类型 | 路径 | 状态 |
| --- | --- | --- |
| 配置模板 | [heatmaps/configs](heatmaps/configs) | 已有 |
| SB/MB 旧版热图结果 | [heatmaps/heatmap_production_results](heatmaps/heatmap_production_results) | 已有 |
| 热图原始结果 | [heatmaps/heatmap_raw_results](heatmaps/heatmap_raw_results) | 已有 |
| 最新 v3 热图 | 待确认 | 当前未见 |

## 7. 风险点

### 7.1 论文叙事风险

- 若直接把 `morph` 结果当作“真实 COO 分型”结论，可能存在标签边界表述风险
- 若把 `all` 的失败或不稳定写得不清楚，论文会显得结论不稳
- 若混用旧版 `s1` 和新版 `v2/v3` 指标，容易造成叙事混乱

### 7.2 实验证据风险

- 最新最优模型缺少成体系的评估与分析目录
- 消融实验未完成，会削弱方法改进的说服力
- 部分文档引用的中间统计文件当前仓库缺失

### 7.3 数据与复现风险

- 原始 WSI 目录当前为空，后续若要补热图或复跑预处理，可能受阻
- `all_uni_features` 使用软链接整合，多机迁移时要注意链接失效风险

### 7.4 时间管理风险

- 如果同时补 `morph`、`nanchang`、`all` 三条线，论文推进容易发散
- 更稳妥的策略是先锁定主线，再决定补哪些支线

## 8. 当前建议结论

当前建议的工作原则：

1. 先把 `morph_uni_clam_sb_v3` 这条线做成“可写论文的完整证据链”
2. 再用 `nanchang` 和 `all` 支撑“稳定性与泛化局限”的讨论
3. 不把未完成实验和探索性结果混入主结论
4. 后续所有新增材料都继续登记到本工作台
