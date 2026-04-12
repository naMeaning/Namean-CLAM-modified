# CLAM (Clustering-constrained Attention Multiple Instance Learning)

CLAM is a high-throughput and interpretable method for whole slide image (WSI) level classification using weakly-supervised learning. It requires only diagnosing slide-level labels and uses an attention-based learning mechanism to identify sub-regions of high diagnostic value to accurately classify whole slides.

## Features

- **Feature Extraction**: Extract features from WSIs using pre-trained computer vision models like ResNet50, UNI, and CONCH.
- **Attention-based MIL**: Utilize clustering-constrained attention multiple instance learning models (CLAM-SB and CLAM-MB) for accurate WSI classification.
- **Interpretability**: The attention mechanism highlights regions in the WSI that contribute most to the classification decision.
- **Flexibility**: Supports various classification tasks including tumor vs. normal and multi-class tumor subtyping.
- **Evaluation Utilities**: Built-in scripts for evaluating trained models and calculating metrics such as Accuracy and ROC AUC.

## Requirements

Ensure you have the following installed:
- PyTorch
- torchvision
- openslide-python
- h5py
- pandas
- numpy
- scikit-learn
- timm
- PIL (Pillow)

Specific dependencies like `conch` or `timm` might be required based on the feature extractor chosen.

## Pre-requisites: Feature Extraction

Before training a model, extract features from WSIs.
You can extract patches and then features or extract them directly using `extract_features.py`:
```bash
python extract_features.py --data_dir /path/to/data --csv_path /path/to/csv --feat_dir /path/to/output_features --model_name resnet50_trunc --batch_size 256
```
Supported models for extraction in `models/builder.py`:
- `resnet50_trunc`
- `uni_v1`
- `conch_v1`
- `conch_v1_5`

*Note: For UNI and CONCH models, you may need to set specific environment variables (`UNI_CKPT_PATH`, `CONCH_CKPT_PATH`) pointing to the model checkpoints.*

## Training

Use `main.py` to train the CLAM models.

Example command for Task 1 (Tumor vs Normal):
```bash
python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code task_1_tumor_vs_normal_100 --task task_1_tumor_vs_normal --model_type clam_sb
```

Example command for Task 2 (Tumor Subtyping):
```bash
python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code task_2_tumor_subtyping_100 --task task_2_tumor_subtyping --model_type clam_mb
```

### Key Training Arguments:

- `--data_root_dir`: Directory containing the extracted features.
- `--model_type`: Choose between `clam_sb` (single-branch), `clam_mb` (multi-branch), or `mil`.
- `--task`: `task_1_tumor_vs_normal` or `task_2_tumor_subtyping`.
- `--bag_loss`: Slide-level classification loss function (`ce` or `svm`).
- `--inst_loss`: Instance-level clustering loss function (`svm` or `ce`).
- `--k`: Number of folds for cross-validation.

Results, including checkpoints and summary statistics, will be saved in the `./results` directory.

## Evaluation

Use `eval.py` to evaluate trained models.

```bash
python eval.py --drop_out 0.25 --models_exp_code task_1_tumor_vs_normal_100_s1 --save_exp_code task_1_tumor_vs_normal_eval --task task_1_tumor_vs_normal --model_type clam_sb
```
Evaluation results and CSV summaries will be saved in `./eval_results`.

## Architectures

The framework supports several Multiple Instance Learning (MIL) architectures found in `models/model_clam.py` and `models/model_mil.py`:
- **CLAM_SB**: CLAM with a single attention branch.
- **CLAM_MB**: CLAM with multiple attention branches (one for each class).
- **MIL_fc**: Standard MIL network with a fully connected bag classifier.
- **MIL_fc_mc**: Multi-class standard MIL network.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3. See `LICENSE.md` for details.
# Namean-CLAM-modified
