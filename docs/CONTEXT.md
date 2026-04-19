# Claude Code 上下文提示词

当你需要让 Claude Code 理解本项目上下文时，发送以下内容：

---

## 上下文模板

```
请阅读以下项目上下文，了解背景后再开始工作：

## 项目概述
CLAM（Clustering-constrained Attention Multiple Instance Learning）用于全切片图像（WSI）分类，目前主要用于 DLBCL 亚型分类（GCB vs non-GCB）。

## 关键数据集
| 数据集 | 患者数 | 特征 | 说明 |
|--------|--------|------|------|
| nanchang | 50 | UNI/ResNet | 单院数据 |
| morph | 132 | UNI/ResNet | Hans算法推导标签 |
| tcga | 39 | UNI/ResNet | TCGA数据库 |
| all | 221 | UNI/ResNet | 三者混合 |

## 当前最佳结果
- **morph_uni_clam_sb_v3**: Test AUC = 0.7806（排除异常fold后 0.86）
- 现有结果目录记录的配置: PCA(256) + CosineAnnealing + 特征增强 + warmup(10 epochs)
- 注意: 若旧实验目录缺少 `s_<fold>_pca.pkl`，评估前需要先回填 PCA 模型或重跑实验

## 已完成实验
- `morph_uni_clam_sb_v1`, `v2`, `v3`
- `morph_uni_clam_sb_strong_v1`

## 待做消融实验（优先级P0）
1. MIL baseline (`--model_type mil`)
2. 纯 bag loss (`--bag_weight 1.0 --no_inst_cluster`)
3. 关闭 PCA (去掉 `--use_pca`)

## 已知问题
- **聚类反转**: 约10-40%的fold会出现，评估时加 `--auto_fix_inversion`
- **过拟合**: morph数据集小，训练误差接近0但验证不稳定
- **方差大**: 每fold测试集仅14例，单折AUC波动大(0.42~0.97)

## 核心文档
- [CLAUDE.md](../CLAUDE.md) - 项目配置说明
- [docs/训练效果问题分析与改进建议.md](../docs/训练效果问题分析与改进建议.md) - 问题分析
- [docs/过拟合专项优化方案.md](../docs/过拟合专项优化方案.md) - 过拟合方案
- [docs/消融实验执行方案.md](../docs/消融实验执行方案.md) - 当前正式消融实验标准

## 训练常用命令模板
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task task_3_dlbcl_coo --dataset morph --feature_type uni \
    --exp_code <实验名> --model_type clam_sb \
    --data_root_dir features --results_dir results \
    --use_pca --pca_dim 256 \
    --early_stopping --monitor_metric val_auc \
    --save_best_auc_ckpt --log_data
```

## 评估命令模板
```bash
python eval.py --task task_3_dlbcl_coo --dataset morph \
    --models_exp_code <exp_code>_s1 --save_exp_code <exp_code>_eval \
    --model_type clam_sb --auto_fix_inversion --ckpt_type auc \
    --data_root_dir features
```

## 训练产物约定
- `s_<fold>_pca.pkl`: fold 级 PCA 模型，`--use_pca` 评估必需
- `artifacts.json`: 每个 fold 的产物清单
- `<fold>/events.out.tfevents.*`: TensorBoard 日志
- `split_<fold>_results.pkl`: 预测结果缓存，不是 PCA 模型
```

---

## 使用方法

### 方法1：快捷指令
直接复制上方模板内容发送给 Claude Code。

### 方法2：引用文档
```
请先阅读 docs/CLAUDE_CODE_CONTEXT.md 了解项目上下文。
```

### 适用场景
- 开始新对话时
- Claude Code 回复偏离主题时
- 需要快速切换任务上下文时
- 解决完问题后继续工作时
