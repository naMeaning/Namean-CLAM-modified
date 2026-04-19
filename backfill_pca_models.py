from __future__ import print_function

import argparse
import os

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.file_utils import save_pkl
from utils.pca_utils import fit_pca_from_train_split


def build_dataset(args):
    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes = 2
        return Generic_MIL_Dataset(
            csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
            data_dir=os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
            patient_strat=False,
            ignore=[],
        )

    if args.task == 'task_2_tumor_subtyping':
        args.n_classes = 3
        return Generic_MIL_Dataset(
            csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
            data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
            patient_strat=False,
            ignore=[],
        )

    if args.task == 'task_3_dlbcl_coo':
        args.n_classes = 2
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
        else:
            raise NotImplementedError(f'Unsupported dataset: {args.dataset}')

        return Generic_MIL_Dataset(
            csv_path=csv_path,
            data_dir=data_dir,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'GCB': 0, 'non-GCB': 1},
            patient_strat=True,
            ignore=[],
        )

    raise NotImplementedError(f'Unsupported task: {args.task}')


def main():
    parser = argparse.ArgumentParser(description='Backfill fold-specific PCA models for existing experiments')
    parser.add_argument('--task', type=str, required=True,
                        choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'task_3_dlbcl_coo'])
    parser.add_argument('--dataset', type=str, default='nanchang',
                        choices=['nanchang', 'tcga', 'morph', 'all'])
    parser.add_argument('--feature_type', type=str, default='resnet',
                        choices=['resnet', 'uni'])
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--models_exp_code', type=str, required=True,
                        help='existing experiment directory name, e.g. morph_uni_clam_sb_v3_s1')
    parser.add_argument('--splits_dir', type=str, default=None,
                        help='optional splits directory; defaults to results_dir/models_exp_code')
    parser.add_argument('--pca_dim', type=int, default=256)
    parser.add_argument('--pca_whiten', action='store_true', default=False)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    args.models_dir = os.path.join(args.results_dir, args.models_exp_code)
    if args.splits_dir is None:
        args.splits_dir = args.models_dir

    assert os.path.isdir(args.models_dir), f'models_dir not found: {args.models_dir}'
    assert os.path.isdir(args.splits_dir), f'splits_dir not found: {args.splits_dir}'

    dataset = build_dataset(args)

    if args.fold >= 0:
        folds = range(args.fold, args.fold + 1)
    else:
        start = 0 if args.k_start == -1 else args.k_start
        end = args.k if args.k_end == -1 else args.k_end
        folds = range(start, end)

    for fold in folds:
        pca_path = os.path.join(args.models_dir, f's_{fold}_pca.pkl')
        if os.path.isfile(pca_path) and not args.overwrite:
            print(f'[skip] fold {fold}: PCA already exists -> {pca_path}')
            continue

        split_csv = os.path.join(args.splits_dir, f'splits_{fold}.csv')
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f'split file not found for fold {fold}: {split_csv}')

        train_split, _, _ = dataset.return_splits(from_id=False, csv_path=split_csv)
        pca_model = fit_pca_from_train_split(
            train_split,
            n_components=args.pca_dim,
            whiten=args.pca_whiten,
        )
        save_pkl(pca_path, pca_model)
        print(f'[ok] fold {fold}: saved PCA model -> {pca_path}')


if __name__ == '__main__':
    main()
