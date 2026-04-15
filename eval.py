# =============================================================================
# 文件功能：独立评估脚本（与训练分离，可对已训练模型进行 K-Fold 批量评估）
# 流程：
#   1. 解析参数（模型路径、任务类型、Fold 范围等）
#   2. 加载对应数据集
#   3. 从 splits_dir 读取每个 Fold 的数据划分
#   4. 逐 Fold 加载 checkpoint → 推理 → 计算 AUC/Acc
#   5. 汇总结果保存为 summary.csv 和每个 Fold 的 fold_<i>.csv
# =============================================================================

from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
from utils.file_utils import load_pkl

# =============================================================================
# 命令行参数解析
# =============================================================================
# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
# 模型 checkpoint 所在目录（results_dir/models_exp_code）
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
# 评估结果保存目录（eval_results/EVAL_<save_exp_code>）
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
# 允许使用与模型不同的 splits（跨数据集评估场景）
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
# K-Fold 范围控制（支持单 Fold 评估 --fold）
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
# AUC 计算模式：micro_average（展平 OvR）vs macro_average（对各类 AUC 取均值）
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
# 评估所用数据分割（train/val/test/all）
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping','task_3_dlbcl_coo'])
parser.add_argument('--dataset', type=str, default='nanchang',
                    choices=['nanchang', 'tcga', 'morph', 'all'],
                    help='选择DLBCL数据集 (仅task_3_dlbcl_coo有效)')
parser.add_argument('--feature_type', type=str, default='resnet',
                    choices=['resnet', 'uni'],
                    help='选择特征类型: resnet 或 uni')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--use_pca', action='store_true', default=False,
                    help='Enable PCA dimensionality reduction (must match training)')
parser.add_argument('--pca_dim', type=int, default=256,
                    help='PCA dimensions (must match training)')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--auto_fix_inversion', action='store_true', default=False,
                    help='自动检测并修正聚类反转问题 (CLAM特有问题)')
parser.add_argument('--ckpt_type', type=str, choices=['default', 'auc', 'loss'], default='default',
                    help='checkpoint类型: default(标准), auc(best_val_auc), loss(best_val_loss)')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建评估结果目录和模型目录路径
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

# 若未指定 splits_dir，默认使用模型同目录下的 splits
if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

# 记录当前评估实验配置到文本文件
settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size,
            'use_pca': args.use_pca,
            'pca_dim': args.pca_dim}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

# 根据任务类型加载对应 MIL 数据集（与 main.py 中保持一致）
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_dlbcl_coo':
    args.n_classes = 2
    # 根据 --dataset 和 --feature_type 参数选择CSV和特征目录
    if args.dataset == 'nanchang':
        csv_path = 'dataset_csv/nanchang_dlbcl.csv'
        data_dir = os.path.join(args.data_root_dir, 'nanchang_{}_features'.format(args.feature_type))
    elif args.dataset == 'tcga':
        csv_path = 'dataset_csv/tcga_dlbcl.csv'
        data_dir = os.path.join(args.data_root_dir, 'tcga_{}_features'.format(args.feature_type))
    elif args.dataset == 'morph':
        csv_path = 'dataset_csv/dlbcl_morph.csv'
        data_dir = os.path.join(args.data_root_dir, 'morph_{}_features'.format(args.feature_type))
    elif args.dataset == 'all':
        csv_path = 'dataset_csv/dlbcl_all.csv'
        data_dir = os.path.join(args.data_root_dir, 'all_{}_features'.format(args.feature_type))

    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=data_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'GCB': 0, 'non-GCB': 1},
        patient_strat=True,
        ignore=[]
    )

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

# 解析 Fold 范围：支持全部 Fold、部分范围、单个 Fold
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

# 构建每个 Fold 对应的 checkpoint 路径列表
# 根据 ckpt_type 选择不同类型的 checkpoint
if args.ckpt_type == 'auc':
    ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint_auc.pt'.format(fold)) for fold in folds]
elif args.ckpt_type == 'loss':
    ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint_loss.pt'.format(fold)) for fold in folds]
else:  # default
    ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]

# 记录实际使用的 checkpoint 类型
settings.update({'ckpt_type': args.ckpt_type})
# 数据集 split 名称到 return_splits 返回值索引的映射（all=-1 表示使用整个数据集）
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            # 使用完整数据集（all 模式）
            split_dataset = dataset
        else:
            # 读取对应 Fold 的 CSV 划分，取指定 split（train/val/test）
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        # PCA 实验的独立评估必须复用训练时同 fold 的 PCA 模型
        if args.use_pca:
            pca_path = os.path.join(args.models_dir, 's_{}_pca.pkl'.format(folds[ckpt_idx]))
            if not os.path.isfile(pca_path):
                raise FileNotFoundError(
                    f"PCA model not found for fold {folds[ckpt_idx]}: {pca_path}. "
                    "Please make sure training saved fold-specific PCA models."
                )
            split_dataset.pca_model = load_pkl(pca_path)
            split_dataset.pca_dim = args.pca_dim

        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        # 每个 Fold 的预测结果保存为独立 CSV
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    # 汇总所有 Fold 的 AUC 和 Accuracy
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
