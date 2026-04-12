# =============================================================================
# 文件功能：模型评估工具函数（独立评估流程，对应 eval.py 使用）
# 主要函数：
#   - initiate_model() : 从 checkpoint 加载并初始化模型
#   - eval()           : 单次完整评估入口
#   - summary()        : 推理 + AUC/错误率汇总，支持多分类 micro average
# =============================================================================

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    """
    从 checkpoint 文件加载并初始化指定模型。
    
    注意：checkpoint 中可能包含 'instance_loss_fn' 相关的 key，
    这些 key 属于损失函数对象（不可序列化），在加载前需要过滤掉，
    以确保 strict=True 的 load_state_dict 能够正常运行。
    同时将 '.module' 前缀去除（应对 DataParallel 保存的权重）。
    """
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        # 过滤掉 instance_loss_fn 相关 key（损失函数对象无法直接加载）
        if 'instance_loss_fn' in key:
            continue
        # 去除 DataParallel 保存时添加的 '.module' 前缀
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    """单次评估入口：加载模型 -> 构建 loader -> 执行 summary -> 打印结果。"""
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    """
    对给定数据集执行完整推理并计算评估指标。
    
    返回：
        patient_results : dict，每个 slide 的预测概率和真实标签
        test_error      : 错误率（1 - 准确率）
        auc_score       : ROC AUC（二分类取正类概率，多分类支持 macro/micro）
        df              : 结果 DataFrame（含 slide_id, Y, Y_hat, p_0, p_1 ...）
        acc_logger      : Accuracy_Logger 对象，用于后续分析
    """
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        # 只有一个类别时无法计算 AUC，返回 -1 标记为无效
        auc_score = -1

    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])

            # =========================================================
            # 检测并修正聚类反转问题
            # 如果 AUC 很低（< 0.3）但翻转后很高（> 0.7），说明发生了聚类反转
            # 聚类反转：cluster 0 对应 class 1，cluster 1 对应 class 0
            # =========================================================
            if hasattr(args, 'auto_fix_inversion') and args.auto_fix_inversion:
                inverted_probs = np.column_stack([all_probs[:, 1], all_probs[:, 0]])
                inverted_auc = roc_auc_score(all_labels, inverted_probs[:, 1])

                if auc_score < 0.3 and inverted_auc > 0.7:
                    print(f"  [WARNING] 检测到聚类反转! AUC: {auc_score:.4f} -> 修正为: {inverted_auc:.4f}")
                    # 翻转预测：Y_hat 和概率都需要翻转
                    all_probs = inverted_probs
                    all_preds = 1 - all_preds  # 翻转预测标签
                    auc_score = inverted_auc  # 更新 AUC
                    # 重新计算错误率
                    test_error = 1 - np.mean(all_preds == all_labels)
        else:
            # 多分类：逐类计算 AUC（OvR 策略）
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                # micro average：将所有类别的预测展平后统一计算一条 ROC 曲线
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                # macro average：对各类别 AUC 取 nanmean
                auc_score = np.nanmean(np.array(aucs))

    # 拼装结果 DataFrame：slide_id + 真实标签 + 预测标签 + 各类别概率
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
