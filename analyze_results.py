# =============================================================================
# 文件功能：分析 CLAM 模型评估结果
# 输入：eval.py 输出的 eval_results/EVAL_xxx_eval/ 目录
# 输出：统计信息、分类报告、混淆矩阵
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import argparse

def analyze_fold(fold_path):
    """
    分析单个 fold 的评估结果

    参数:
        fold_path: fold_X.csv 文件路径

    返回:
        dict: 包含 AUC、准确率、预测结果等
    """
    df = pd.read_csv(fold_path)

    # 真实标签 (Y) 和预测标签 (Y_hat)
    y_true = df['Y'].values
    y_pred = df['Y_hat'].values

    # 预测概率 (p_0=non-GCB, p_1=GCB)
    y_prob = df['p_1'].values  # 使用 GC B 的概率

    # 计算指标
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=['non-GCB', 'GCB'])

    return {
        'auc': auc,
        'acc': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'n_samples': len(df),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def analyze_all_folds(eval_dir, output_dir=None):
    """
    分析所有 fold 的结果，并计算汇总统计

    参数:
        eval_dir: eval_results/EVAL_xxx_eval/ 目录路径
        output_dir: 可选，保存分析结果的目录

    返回:
        dict: 包含各 fold 统计和总体统计
    """
    summary_path = os.path.join(eval_dir, 'summary.csv')
    summary_df = pd.read_csv(summary_path)

    print("=" * 60)
    print("CLAM 模型评估结果分析")
    print("=" * 60)

    # 各 fold 结果
    fold_results = []
    for i in range(10):
        fold_path = os.path.join(eval_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            result = analyze_fold(fold_path)
            result['fold'] = i
            fold_results.append(result)
            print(f"\nFold {i}:")
            print(f"  AUC: {result['auc']:.4f}")
            print(f"  Accuracy: {result['acc']:.4f}")
            print(f"  样本数: {result['n_samples']}")

    # 汇总统计
    aucs = [r['auc'] for r in fold_results]
    accs = [r['acc'] for r in fold_results]

    print("\n" + "=" * 60)
    print("汇总统计 (All Folds)")
    print("=" * 60)
    print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  各折 AUC: {[f'{x:.4f}' for x in aucs]}")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  各折 Accuracy: {[f'{x:.4f}' for x in accs]}")

    # 保存汇总结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 保存汇总CSV
        summary_stats = pd.DataFrame({
            'fold': range(len(aucs)),
            'auc': aucs,
            'accuracy': accs
        })
        summary_stats.loc['mean'] = ['mean', np.mean(aucs), np.mean(accs)]
        summary_stats.loc['std'] = ['std', np.std(aucs), np.std(accs)]
        summary_stats.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)

        # 保存分类报告
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("分类报告 (Classification Report)\n")
            f.write("=" * 60 + "\n\n")

            for result in fold_results:
                f.write(f"\n--- Fold {result['fold']} ---\n")
                f.write(result['classification_report'])

            # 总体混淆矩阵
            f.write("\n" + "=" * 60 + "\n")
            f.write("总体混淆矩阵 (Aggregated Confusion Matrix)\n")
            f.write("=" * 60 + "\n\n")

            # 合并所有 fold 的预测结果
            all_y_true = np.concatenate([r['y_true'] for r in fold_results])
            all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

            overall_cm = confusion_matrix(all_y_true, all_y_pred)
            f.write(f"              non-GCB  GCB\n")
            f.write(f"non-GCB      {overall_cm[0,0]:5d}  {overall_cm[0,1]:5d}\n")
            f.write(f"GCB          {overall_cm[1,0]:5d}  {overall_cm[1,1]:5d}\n")

        print(f"\n分析结果已保存到: {output_dir}")

    return {
        'fold_results': fold_results,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs)
    }


def compare_experiments(eval_dirs, experiment_names=None):
    """
    对比多个实验的结果

    参数:
        eval_dirs: 多个 eval_results 目录路径的列表
        experiment_names: 对应的实验名称列表

    返回:
        DataFrame: 对比表格
    """
    if experiment_names is None:
        experiment_names = [f"Exp_{i}" for i in range(len(eval_dirs))]

    results = []
    for eval_dir, name in zip(eval_dirs, experiment_names):
        summary_path = os.path.join(eval_dir, 'summary.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            aucs = df['test_auc'].values
            accs = df['test_acc'].values
            results.append({
                'Experiment': name,
                'Mean AUC': np.mean(aucs),
                'Std AUC': np.std(aucs),
                'Mean Acc': np.mean(accs),
                'Std Acc': np.std(accs),
                'Min AUC': np.min(aucs),
                'Max AUC': np.max(aucs)
            })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("多实验对比 (Experiment Comparison)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析 CLAM 模型评估结果')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='eval_results 目录路径, 如 eval_results/EVAL_dlbcl_all_uni_clam_sb_eval')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='分析结果输出目录')
    parser.add_argument('--compare', action='store_true',
                        help='是否对比多个实验')

    args = parser.parse_args()

    if args.compare:
        # 多实验对比模式
        print("多实验对比模式需要手动指定实验路径")
        print("示例用法:")
        print("  python analyze_results.py \\")
        print("    --compare \\")
        print("    --eval_dirs eval_results/EVAL_exp1_eval eval_results/EVAL_exp2_eval \\")
        print("    --experiment_names exp1 exp2")
    else:
        # 单实验分析
        analyze_all_folds(args.eval_dir, args.output_dir)
