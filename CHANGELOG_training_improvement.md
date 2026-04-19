# CLAM 训练效果改进修改日志

本文档记录每次对项目训练流程的修改，便于追踪和回溯。

---

## [2026-04-13] 训练效果改进 - 基础架构

### 修改文件
- `dataset_csv/dlbcl_all.csv`
- `utils/core_utils.py`
- `utils/utils.py`
- `dataset_modules/dataset_generic.py`
- `main.py`
- `create_splits_seq.py`
- `eval.py`
- `CLAUDE.md`
- `operation_guide.md`
- `CHANGELOG_training_improvement.md`

### 修改内容

#### 1. dlbcl_all.csv 添加 source 列
- 为 all 数据集 CSV 添加 `source` 列，标识样本来源（tcga/nanchang/morph）
- 便于做 source-aware split 和分层分析

#### 2. EarlyStopping 增强 (utils/core_utils.py)
- 新增 `mode` 参数：支持 `'min'` (val_loss) 和 `'max'` (val_auc)
- 新增 `monitor_metric` 属性记录当前监控指标
- `__call__` 方法支持两种监控模式，同时跟踪 best_loss 和 best_auc
- 新增 `get_best_metrics()` 方法返回最佳指标信息
- 支持同时保存 best_auc 和 best_loss 两套 checkpoint

#### 3. train() 函数改造 (utils/core_utils.py)
- validate/validate_clam 返回 (val_loss, val_auc) 两个指标
- 当 args.save_best_auc_ckpt=True 时，同时保存 s_{fold}_checkpoint_auc.pt 和 s_{fold}_checkpoint_loss.pt
- 记录最佳 epoch 信息用于后续分析

#### 4. generate_split 增强 (utils/utils.py)
- 新增 `source_aware` 参数
- 当启用时，按 (label, source) 组合分层采样

#### 5. dataset_generic.py 增强
- `Generic_WSI_Classification_Dataset.__init__` 新增 `source_col` 参数
- `patient_data_prep` 支持提取 source 信息
- `cls_ids_prep` 支持创建 source_aware_cls_ids
- `create_splits` 支持 source-aware 划分

#### 6. main.py 新增参数
- `--monitor_metric {val_auc,val_loss}`: 早停监控指标，默认 val_auc (DLBCL任务)
- `--save_best_auc_ckpt`: 保存 best_auc checkpoint
- DLBCL 任务默认参数调整：drop_out=0.5, reg=1e-3, weighted_sample=False
- Nanchang 数据集默认 5 折（原10折）

#### 7. create_splits_seq.py source-aware split
- all 数据集默认启用 source-aware split（通过 source_col='source'）
- 使用 (label, source) 组合做分层，确保每个 fold 的 source 分布均衡

#### 8. eval.py 新增参数
- `--ckpt_type {default,auc,loss}`: 指定加载哪种 checkpoint
- default 模式兼容旧的 checkpoint 命名

#### 9. 文档更新
- CLAUDE.md: 新增参数说明和 DLBCL 推荐配置
- operation_guide.md: 新增章节说明 source-aware split 和 checkpoint 类型

### 影响范围
- 训练流程：早停机制、checkpoint 保存策略
- 评估流程：支持选择不同类型的 checkpoint
- 数据划分：all 数据集 source 分布更均衡
- 默认参数：DLBCL 任务更保守

### 相关文档
- `docs/训练效果问题分析与改进建议.md` - 问题分析与改进方案原文
