# =============================================================================
# 文件功能：K-Fold 数据划分脚本
# 用途：按 patient-level 将数据集划分为 train/val/test 三组，
#       支持多个 label_frac（半监督场景），生成以下文件：
#         - splits_<i>.csv        : 各 Fold 的 slide_id 列表
#         - splits_<i>_bool.csv   : one-hot 格式的划分标志
#         - splits_<i>_descriptor.csv : 各分组各类别样本数统计
# =============================================================================

import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
# 训练标签比例（< 0 时自动遍历 [0.1, 0.25, 0.5, 0.75, 1.0] 五个比例）
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping','task_3_dlbcl_coo'])
parser.add_argument('--dataset', type=str, default='nanchang',
                    choices=['nanchang', 'tcga', 'morph', 'all'],
                    help='选择DLBCL数据集 (仅task_3_dlbcl_coo有效)')
# val_frac 和 test_frac 为各类别数量的比例（按患者数量计算）
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

# 根据任务类型加载对应数据集（patient_strat=True 表示按患者级别划分，避免同一患者的不同 slide 跨入 train/test）
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',   # 亚型分类使用多数投票决定患者标签
                            ignore=[])
elif args.task == 'task_3_dlbcl_coo':
    args.n_classes = 2
    # 根据 --dataset 参数选择CSV（支持 nanchang/tcga/morph/all）
    csv_path = 'dataset_csv/nanchang_dlbcl.csv'  # 默认
    if hasattr(args, 'dataset') and args.dataset:
        if args.dataset == 'nanchang':
            csv_path = 'dataset_csv/nanchang_dlbcl.csv'
        elif args.dataset == 'tcga':
            csv_path = 'dataset_csv/tcga_dlbcl.csv'
        elif args.dataset == 'morph':
            csv_path = 'dataset_csv/dlbcl_morph.csv'
        elif args.dataset == 'all':
            csv_path = 'dataset_csv/dlbcl_all.csv'
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=csv_path,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'GCB': 0, 'non-GCB': 1},
        patient_strat=True,
        ignore=[]
    )


else:
    raise NotImplementedError

# 按各类别患者数量的比例计算 val/test 绝对样本数
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        # 单一 label_frac：只生成一套划分
        label_fracs = [args.label_frac]
    else:
        # label_frac <= 0：批量生成五种半监督比例的划分
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        # 每个 label_frac 对应独立的输出目录
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            # 保存标准格式和 bool 格式两种 CSV，以及各类别统计表
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))




