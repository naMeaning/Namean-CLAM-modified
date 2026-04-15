# 文档与目录导航

> 最后更新：2026-04-15
> 用途：提供当前仓库中文档与关键目录的导航入口。

## 1. 当前优先阅读顺序

如果目的是完成毕业论文，建议按以下顺序阅读：

1. [project_understanding.md](project_understanding.md)
2. [ongoing_thesis_workspace.md](ongoing_thesis_workspace.md)
3. [operation_guide.md](operation_guide.md)
4. [sdpc_modifications.md](sdpc_modifications.md)
5. [docs/消融实验执行方案.md](docs/消融实验执行方案.md)

## 2. 核心文档

| 文档 | 作用 |
| --- | --- |
| [project_understanding.md](project_understanding.md) | 项目理解总览，适合作为论文背景底稿 |
| [ongoing_thesis_workspace.md](ongoing_thesis_workspace.md) | 论文持续工作台，适合长期更新 |
| [operation_guide.md](operation_guide.md) | 当前推荐执行流程 |
| [sdpc_modifications.md](sdpc_modifications.md) | `.sdpc` 与 DLBCL 定制说明 |
| [docs/消融实验执行方案.md](docs/消融实验执行方案.md) | 当前正式消融实验执行标准 |
| [docs/README_Chinese.md](docs/README_Chinese.md) | 中文文档索引 |

## 3. 历史分析与参考文档

| 文档 | 当前定位 |
| --- | --- |
| [docs/训练效果问题分析与改进建议.md](docs/训练效果问题分析与改进建议.md) | 深度分析稿，保留参考价值，但需结合当前代码理解 |
| [docs/过拟合专项优化方案.md](docs/过拟合专项优化方案.md) | 过拟合专项分析 |
| [docs/ablation_experiments.md](docs/ablation_experiments.md) | 历史消融草案，正式执行以“消融实验执行方案” 为准 |
| [project_summary.md](project_summary.md) | 方法和流程总结，保留概念价值 |

## 4. 关键代码入口

| 文件 | 作用 |
| --- | --- |
| [main.py](main.py) | 训练入口 |
| [eval.py](eval.py) | 评估入口 |
| [create_splits_seq.py](create_splits_seq.py) | 划分生成 |
| [extract_features_fp.py](extract_features_fp.py) | 特征提取 |
| [create_patches_fp.py](create_patches_fp.py) | patch 坐标提取 |
| [create_heatmaps.py](create_heatmaps.py) | 热图生成 |

## 5. 关键数据目录

| 路径 | 说明 |
| --- | --- |
| [dataset_csv](dataset_csv) | 数据标签表 |
| [features](features) | 训练用特征 |
| [splits](splits) | 交叉验证划分 |
| [results](results) | 训练输出 |
| [eval_results](eval_results) | 独立评估输出 |
| [analysis_results](analysis_results) | 分析结果 |
| [heatmaps](heatmaps) | 热图与可解释性结果 |

## 6. 当前不再作为主导航的信息

以下历史内容不再作为当前主导航依据：

- 不存在的脚本引用，如 `make_clam_csv.py`
- 已废弃的单一 DLBCL CSV 命名
- 旧版 slide-level DLBCL 配置说明

## 7. 当前结论

如果后续还需要扩展文档体系，建议继续保持：

- 一份总览：`project_understanding.md`
- 一份工作台：`ongoing_thesis_workspace.md`
- 一份执行指南：`operation_guide.md`
- 一份专题方案：如消融实验、热图、论文写作
