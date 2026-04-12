# =============================================================================
# 文件功能：可视化 CLAM 模型评估结果
# 输入：eval.py 输出的 eval_results/EVAL_xxx_eval/ 目录
# 输出：ROC曲线、混淆矩阵、AUC分布图、预测概率分布图等
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    auc
)
from scipy import stats
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def plot_roc_curves(eval_dir, output_dir, experiment_name=None):
    """
    绘制各 fold 的 ROC 曲线和平均 ROC 曲线

    参数:
        eval_dir: eval_results 目录
        output_dir: 输出目录
        experiment_name: 实验名称（用于标题和文件名）
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    all_y_true = []
    all_y_prob = []
    fold_aucs = []

    for i in range(10):
        fold_path = os.path.join(eval_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            y_true = df['Y'].values
            y_prob = df['p_1'].values

            all_y_true.extend(y_true)
            all_y_prob.extend(y_prob)

            # 计算该 fold 的 ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fold_auc = roc_auc_score(y_true, y_prob)
            fold_aucs.append(fold_auc)

            # 绘制各 fold 的 ROC 曲线（淡色）
            ax.plot(fpr, tpr, alpha=0.3, color='gray', linewidth=1)

    # 绘制平均 ROC 曲线
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    mean_fpr = np.linspace(0, 1, 100)

    # 插值计算平均
    tprs = []
    for i in range(10):
        fold_path = os.path.join(eval_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            fpr, tpr, _ = roc_curve(df['Y'].values, df['p_1'].values)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    ax.plot(mean_fpr, mean_tpr, color='darkorange', linewidth=2,
            label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    # 绘制随机分类器对角线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'{experiment_name}\nROC Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存: {os.path.join(output_dir, 'roc_curve.png')}")


def plot_confusion_matrix(eval_dir, output_dir, experiment_name=None):
    """
    绘制聚合的混淆矩阵

    参数:
        eval_dir: eval_results 目录
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    # 合并所有 fold 的预测结果
    all_y_true = []
    all_y_pred = []

    for i in range(10):
        fold_path = os.path.join(eval_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            all_y_true.extend(df['Y'].values)
            all_y_pred.extend(df['Y_hat'].values)

    cm = confusion_matrix(all_y_true, all_y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['non-GCB', 'GCB'],
                yticklabels=['non-GCB', 'GCB'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{experiment_name}\nConfusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {os.path.join(output_dir, 'confusion_matrix.png')}")


def plot_auc_distribution(eval_dir, output_dir, experiment_name=None):
    """
    绘制各 fold AUC 的分布图和箱线图

    参数:
        eval_dir: eval_results 目录
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    summary_path = os.path.join(eval_dir, 'summary.csv')
    df = pd.read_csv(summary_path)
    aucs = df['test_auc'].values
    accs = df['test_acc'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUC 分布
    ax1 = axes[0]
    ax1.bar(range(len(aucs)), aucs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=np.mean(aucs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aucs):.4f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title(f'{experiment_name}\nAUC per Fold', fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Accuracy 分布
    ax2 = axes[1]
    ax2.bar(range(len(accs)), accs, color='seagreen', alpha=0.7, edgecolor='black')
    ax2.axhline(y=np.mean(accs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accs):.4f}')
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{experiment_name}\nAccuracy per Fold', fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"AUC 分布图已保存: {os.path.join(output_dir, 'auc_distribution.png')}")


def plot_probability_distribution(eval_dir, output_dir, experiment_name=None):
    """
    绘制预测概率分布图，按真实标签分组

    参数:
        eval_dir: eval_results 目录
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    # 合并所有 fold
    all_data = []
    for i in range(10):
        fold_path = os.path.join(eval_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            df['fold'] = i
            all_data.append(df)

    all_df = pd.concat(all_data, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # non-GCB 样本的预测概率分布
    ax1 = axes[0]
    non_gcb_probs = all_df[all_df['Y'] == 0]['p_1'].values  # non-GCB 被预测为 GCB 的概率
    ax1.hist(non_gcb_probs, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold: 0.5')
    ax1.set_xlabel('Predicted Probability (GCB)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{experiment_name}\nProbability Distribution (True: non-GCB)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # GCB 样本的预测概率分布
    ax2 = axes[1]
    gcb_probs = all_df[all_df['Y'] == 1]['p_1'].values  # GCB 被预测为 GCB 的概率
    ax2.hist(gcb_probs, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold: 0.5')
    ax2.set_xlabel('Predicted Probability (GCB)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'{experiment_name}\nProbability Distribution (True: GCB)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"概率分布图已保存: {os.path.join(output_dir, 'probability_distribution.png')}")


def plot_boxplot_comparison(eval_dirs, experiment_names, output_dir):
    """
    绘制多个实验的 AUC/Accuracy 箱线图对比

    参数:
        eval_dirs: 多个 eval_results 目录路径列表
        experiment_names: 实验名称列表
        output_dir: 输出目录
    """
    all_aucs = []
    all_accs = []

    for eval_dir in eval_dirs:
        summary_path = os.path.join(eval_dir, 'summary.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            all_aucs.append(df['test_auc'].values)
            all_accs.append(df['test_acc'].values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUC 箱线图
    ax1 = axes[0]
    bp1 = ax1.boxplot(all_aucs, labels=experiment_names, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiment_names)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC Comparison (10-Fold CV)', fontsize=14)
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # Accuracy 箱线图
    ax2 = axes[1]
    bp2 = ax2.boxplot(all_accs, labels=experiment_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Comparison (10-Fold CV)', fontsize=14)
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比箱线图已保存: {os.path.join(output_dir, 'comparison_boxplot.png')}")


def visualize_all(eval_dir, output_dir=None, experiment_name=None):
    """
    生成所有可视化图表

    参数:
        eval_dir: eval_results 目录路径
        output_dir: 输出目录，默认在 eval_dir 下创建 visualization 子目录
        experiment_name: 实验名称（用于标题）
    """
    if output_dir is None:
        output_dir = os.path.join(eval_dir, 'visualization')

    os.makedirs(output_dir, exist_ok=True)

    if experiment_name is None:
        experiment_name = os.path.basename(eval_dir).replace('EVAL_', '').replace('_eval', '')

    print("=" * 60)
    print(f"生成可视化结果: {experiment_name}")
    print("=" * 60)

    # 1. ROC 曲线
    plot_roc_curves(eval_dir, output_dir, experiment_name)

    # 2. 混淆矩阵
    plot_confusion_matrix(eval_dir, output_dir, experiment_name)

    # 3. AUC/Accuracy 分布
    plot_auc_distribution(eval_dir, output_dir, experiment_name)

    # 4. 预测概率分布
    plot_probability_distribution(eval_dir, output_dir, experiment_name)

    print("\n所有可视化结果已保存到:", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化 CLAM 模型评估结果')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='eval_results 目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认创建 visualization 子目录')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称（用于图表标题）')
    parser.add_argument('--compare', action='store_true',
                        help='对比模式：同时绘制多个实验')
    parser.add_argument('--eval_dirs', nargs='+', default=[],
                        help='对比模式的多个 eval_dirs')
    parser.add_argument('--experiment_names', nargs='+', default=[],
                        help='对比模式的实验名称')

    args = parser.parse_args()

    if args.compare:
        # 多实验对比模式
        if args.eval_dirs and args.experiment_names:
            os.makedirs(args.output_dir or 'comparison_plots', exist_ok=True)
            plot_boxplot_comparison(
                args.eval_dirs,
                args.experiment_names,
                args.output_dir or 'comparison_plots'
            )
        else:
            print("对比模式需要提供 --eval_dirs 和 --experiment_names")
    else:
        # 单实验可视化
        visualize_all(args.eval_dir, args.output_dir, args.name)
