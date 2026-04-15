# =============================================================================
# 文件功能：CLAM 项目训练主入口
# 功能说明：
#   - 解析命令行参数（通用训练参数 + CLAM 专用参数）
#   - 根据任务类型加载对应的 WSI MIL 数据集
#   - 执行 K-Fold 交叉验证训练流程
#   - 汇总所有 Fold 的 AUC / Accuracy 并保存 CSV 结果
# =============================================================================

from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # 若 results_dir 不存在则创建
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # 解析 K-Fold 起止范围（-1 表示使用默认值 0 或 args.k）
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # 用于收集每个 Fold 的测试/验证 AUC 和 Accuracy
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)

    # 构建特征增强配置（传递给 DataLoader）
    args.aug_config = {
        'feature_noise_std': args.feature_noise_std,
        'feature_dropout': args.feature_dropout,
        'patch_keep_ratio': args.patch_keep_ratio,
        'max_patches_per_bag': args.max_patches_per_bag,
    }

    for i in folds:
        seed_torch(args.seed)
        # 根据当前 Fold 的 CSV split 文件加载 train/val/test 数据集
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        # 执行单 Fold 训练，返回结果字典与各评估指标
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # 汇总所有 Fold 结果并保存为 CSV
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    # 如果只跑了部分 Fold，文件名中标注起止范围
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# =============================================================================
# 命令行参数解析
# =============================================================================

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')

# --- 数据与嵌入维度 ---
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)

# --- 训练超参数 ---
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')

# --- K-Fold 交叉验证参数 ---
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')

# --- 结果保存与日志 ---
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

# --- 训练策略 ---
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping','task_3_dlbcl_coo'])
parser.add_argument('--dataset', type=str, default='nanchang',
                    choices=['nanchang', 'tcga', 'morph', 'all'],
                    help='选择DLBCL数据集 (仅task_3_dlbcl_coo有效)')
parser.add_argument('--feature_type', type=str, default='resnet',
                    choices=['resnet', 'uni'],
                    help='选择特征类型: resnet 或 uni')

### CLAM specific options
# CLAM 专用参数：实例级聚类损失配置
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

# --- 特征增强参数 ---
parser.add_argument('--feature_noise_std', type=float, default=0.0,
                    help='Gaussian noise std for feature augmentation (default: 0.0, recommended: 0.02)')
parser.add_argument('--feature_dropout', type=float, default=0.0,
                    help='Feature dimension dropout probability (default: 0.0, recommended: 0.1)')
parser.add_argument('--patch_keep_ratio', type=float, default=1.0,
                    help='Ratio of patches to keep per bag during training (default: 1.0, recommended: 0.8)')
parser.add_argument('--max_patches_per_bag', type=int, default=None,
                    help='Maximum patches per bag (default: None, recommended: 512)')

# --- PCA 降维参数 ---
parser.add_argument('--use_pca', action='store_true', default=False,
                    help='Enable PCA dimensionality reduction')
parser.add_argument('--pca_dim', type=int, default=256,
                    help='PCA dimensions (default: 256, nanchang=256, morph=384, all=512)')
parser.add_argument('--pca_whiten', action='store_true', default=False,
                    help='Whiten PCA transformation')

# --- 训练策略参数 ---
parser.add_argument('--warmup_bag_only_epochs', type=int, default=0,
                    help='First N epochs use bag-loss only, no instance clustering (default: 0, recommended: 10)')
parser.add_argument('--attention_entropy_weight', type=float, default=0.0,
                    help='Attention entropy regularization weight (default: 0.0, recommended: 1e-3)')
parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help='Label smoothing for cross entropy loss (default: 0.0, recommended: 0.05)')

# --- SWA 参数 ---
parser.add_argument('--use_swa', action='store_true', default=False,
                    help='Enable Stochastic Weight Averaging')
parser.add_argument('--swa_start_epoch', type=int, default=10,
                    help='Start SWA from epoch N (default: 10)')
parser.add_argument('--swa_lr', type=float, default=1e-5,
                    help='SWA learning rate (default: 1e-5)')

# --- 早停与模型选择参数 ---
parser.add_argument('--monitor_metric', type=str, choices=['val_auc', 'val_loss'], default='val_auc',
                    help='metric to monitor for early stopping (default: val_auc for DLBCL tasks)')
parser.add_argument('--save_best_auc_ckpt', action='store_true', default=False,
                    help='save both best_auc and best_loss checkpoints')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    """
    固定所有随机种子，保证实验可复现。
    覆盖范围：Python random / NumPy / PyTorch CPU / PyTorch CUDA（单卡和多卡）
    同时关闭 cuDNN 的随机算法优化，改用确定性算法。
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

# =============================================================================
# DLBCL 任务默认参数调整（更保守的配置，减少过拟合）
# =============================================================================
if args.task == 'task_3_dlbcl_coo':
    # DLBCL 任务使用更保守的默认参数
    args.drop_out = 0.5  # 原默认值 0.25，更高的 dropout 减少过拟合
    args.reg = 1e-3      # 原默认值 1e-5，更强的权重衰减
    args.weighted_sample = False  # 原默认 True，但 DLBCL 数据类别接近平衡

    # DLBCL 任务的特征增强默认参数（减少过拟合）
    if args.feature_noise_std == 0.0:
        args.feature_noise_std = 0.02
    if args.feature_dropout == 0.0:
        args.feature_dropout = 0.1
    if args.patch_keep_ratio == 1.0:
        args.patch_keep_ratio = 0.8
    if args.max_patches_per_bag is None:
        args.max_patches_per_bag = 512

    # DLBCL 任务的训练策略默认参数
    if args.warmup_bag_only_epochs == 0:
        args.warmup_bag_only_epochs = 10
    if args.attention_entropy_weight == 0.0:
        args.attention_entropy_weight = 1e-3
    if args.label_smoothing == 0.0:
        args.label_smoothing = 0.05

    # 学习率降低（配合 Cosine Annealing 效果更好）
    if args.lr == 1e-4:  # 仅当使用默认学习率时降低
        args.lr = 5e-5

    # Nanchang 数据集只有 50 例患者，默认 5 折更稳定（原 10 折）
    if args.dataset == 'nanchang' and args.k == 10:
        args.k = 5
        print(f"\n[INFO] Nanchang dataset (50 cases) using 5-fold CV for more stable evaluation")

# 默认特征维度（ResNet-50 截断特征层输出维度）
encoding_size = 1024
# 将当前实验配置打包为字典，用于日志记录
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'monitor_metric': args.monitor_metric,
            'save_best_auc_ckpt': args.save_best_auc_ckpt,
            'feature_noise_std': args.feature_noise_std,
            'feature_dropout': args.feature_dropout,
            'patch_keep_ratio': args.patch_keep_ratio,
            'max_patches_per_bag': args.max_patches_per_bag,
            'use_pca': args.use_pca,
            'pca_dim': args.pca_dim,
            'warmup_bag_only_epochs': args.warmup_bag_only_epochs,
            'attention_entropy_weight': args.attention_entropy_weight,
            'label_smoothing': args.label_smoothing}

# CLAM 模型额外记录 bag_weight、实例损失函数类型和采样数 B
if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

# =============================================================================
# 根据 task 参数加载对应数据集
# task_1_tumor_vs_normal：肿瘤 vs. 正常二分类
# task_2_tumor_subtyping：肿瘤亚型三分类
# =============================================================================

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    # 亚型分类必须启用 subtyping 标志
    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
elif args.task == 'task_3_dlbcl_coo':
    args.n_classes = 2
    # 根据 --dataset 参数选择CSV和特征目录
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
        feat_suffix = 'all_{}_features'.format(args.feature_type)
        data_dir = os.path.join(args.data_root_dir, feat_suffix)

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

else:
    raise NotImplementedError
    
# 创建结果目录（实验代码 + 随机种子 作为子目录名）
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# 推断数据划分目录（splits/<task>_<dataset>_<label_frac*100>）
# task_3_dlbcl_coo 会根据 --dataset 参数自动查找对应目录
if args.split_dir is None:
    if args.task == 'task_3_dlbcl_coo' and hasattr(args, 'dataset'):
        args.split_dir = os.path.join('splits', args.task+'_'+args.dataset+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


# 将实验配置写入文本文件，便于事后溯源
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")



