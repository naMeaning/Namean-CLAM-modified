# =============================================================================
# 文件功能：训练/验证/测试核心工具函数及辅助类
# 主要组件：
#   - Accuracy_Logger  : 分类准确率记录器（支持单样本和批量记录）
#   - EarlyStopping    : 验证集损失驱动的早停机制
#   - train()          : 单 Fold 完整训练流程（模型构建 + 训练循环 + 测试汇总）
#   - train_loop_clam(): CLAM 专用训练循环（bag loss + instance loss 联合优化）
#   - train_loop()     : 通用 MIL 训练循环（仅 bag loss）
#   - validate()       : 通用验证循环（含 AUC 计算）
#   - validate_clam()  : CLAM 专用验证循环（含实例级聚类损失）
#   - summary()        : 测试集完整评估，生成 patient-level 结果字典
# =============================================================================

import numpy as np
import torch
from utils.utils import *
import os
from utils.file_utils import save_pkl
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """分类准确率记录器，按类别分别统计预测正确数与总数。"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        # 每个类别维护 count（总样本数）和 correct（预测正确数）
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        # 记录单个样本的预测结果
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        # 批量记录预测结果（用于实例级评估）
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        # 返回类别 c 的准确率、正确数和总数
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience.

    Supports both loss-based early stopping (mode='min') and AUC-based early stopping (mode='max').
    Can track and save best checkpoint for both loss and AUC metrics simultaneously.
    """
    def __init__(self, patience=20, stop_epoch=50, verbose=False, mode='min', monitor_metric='val_loss'):
        """
        Args:
            patience (int): How long to wait after last improvement. Default: 20
            stop_epoch (int): Earliest epoch possible for stopping. Default: 50
            verbose (bool): If True, prints messages for improvements. Default: False
            mode (str): 'min' for loss-based stopping, 'max' for AUC-based stopping. Default: 'min'
            monitor_metric (str): Name of the metric being monitored. Default: 'val_loss'
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.mode = mode  # 'min' for loss, 'max' for AUC
        self.monitor_metric = monitor_metric

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # For dual checkpoint tracking
        self.best_loss = np.inf
        self.best_loss_epoch = -1
        self.best_auc = 0.0
        self.best_auc_epoch = -1

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt', val_auc=None):
        """
        Check if validation metric improved and decide whether to early stop.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss value
            model: Model to save
            ckpt_name: Checkpoint filename
            val_auc: Validation AUC value (optional, for AUC-based tracking)
        """
        if self.mode == 'max':
            # Monitoring AUC - higher is better
            score = val_auc if val_auc is not None else -val_loss
            if self.best_score is None:
                self.best_score = score
                self.best_auc = val_auc if val_auc is not None else 0.0
                self.best_auc_epoch = epoch
                self.save_checkpoint(model, ckpt_name, 'auc')
            elif score > self.best_score:
                self.best_score = score
                self.best_auc = val_auc if val_auc is not None else 0.0
                self.best_auc_epoch = epoch
                self.save_checkpoint(model, ckpt_name, 'auc')
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
        else:
            # Monitoring loss - lower is better
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.best_loss = val_loss
                self.best_loss_epoch = epoch
                self.save_checkpoint(model, ckpt_name, 'loss')
            elif score > self.best_score:
                self.best_score = score
                self.best_loss = val_loss
                self.best_loss_epoch = epoch
                self.save_checkpoint(model, ckpt_name, 'loss')
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True

    def save_checkpoint(self, model, ckpt_name, metric_name='loss'):
        """Saves model checkpoint with metric name annotation."""
        if self.verbose:
            if metric_name == 'auc':
                print(f'Validation AUC improved ({self.best_auc:.4f}). Saving model to {ckpt_name}...')
            else:
                print(f'Validation loss decreased ({self.best_loss:.6f}). Saving model to {ckpt_name}...')
        torch.save(model.state_dict(), ckpt_name)

    def get_best_metrics(self):
        """Return best loss and AUC info for logging."""
        return {
            'best_loss': self.best_loss,
            'best_loss_epoch': self.best_loss_epoch,
            'best_auc': self.best_auc,
            'best_auc_epoch': self.best_auc_epoch,
            'monitor_mode': self.mode,
            'monitor_metric': self.monitor_metric
        }

def train(datasets, cur, args):
    """   
        train for a single fold
        执行单个 K-Fold 的完整训练流程：
          1. 初始化 TensorBoard writer（可选）
          2. 打印 train/val/test 样本量
          3. 初始化损失函数（CE 或 SVM）
          4. 构建模型（CLAM_SB / CLAM_MB / MIL_fc / MIL_fc_mc）
          5. 构建优化器和 DataLoader
          6. 按 epoch 执行训练和验证循环
          7. 早停或保存最终 checkpoint
          8. 在测试集上执行 summary 并返回结果
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    # 将 train/val/test 划分保存为 CSV，便于复现
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        # 使用 SmoothTop1SVM 作为 bag 级别损失（需安装 topk 库）
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        label_smoothing = getattr(args, 'label_smoothing', 0.0)
        if label_smoothing > 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    # 当启用 PCA 时，使用 PCA 降维后的维度作为模型的 embed_dim
    model_embed_dim = args.pca_dim if getattr(args, 'use_pca', False) else args.embed_dim
    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": model_embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        # 实例级损失函数选择（SVM 或 CrossEntropy）
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    # Cosine Annealing LR 调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    print(f'CosineAnnealingLR: initial_lr={optimizer.param_groups[0]["lr"]}, T_max={args.max_epochs}, eta_min=1e-6')

    print('\nInit Loaders...', end=' ')
    # 构造 aug_config（如果 args 有设置）
    aug_config = getattr(args, 'aug_config', None)

    # PCA 降维（仅在训练集上 fit，避免信息泄漏）
    pca_model = None
    pca_dim = None
    if getattr(args, 'use_pca', False):
        from utils.pca_utils import fit_pca_from_train_split
        pca_dim = getattr(args, 'pca_dim', 256)
        pca_whiten = getattr(args, 'pca_whiten', False)
        pca_model = fit_pca_from_train_split(train_split, n_components=pca_dim, whiten=pca_whiten)
        # 保存每个 fold 对应的 PCA 模型，供独立评估时复用完全相同的变换
        save_pkl(os.path.join(args.results_dir, "s_{}_pca.pkl".format(cur)), pca_model)

    train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                   weighted=args.weighted_sample, aug_config=aug_config,
                                   pca_model=pca_model, pca_dim=pca_dim)
    val_loader = get_split_loader(val_split, testing=args.testing,
                                  pca_model=pca_model, pca_dim=pca_dim)
    test_loader = get_split_loader(test_split, testing=args.testing,
                                   pca_model=pca_model, pca_dim=pca_dim)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        # Determine mode based on monitor_metric
        monitor_metric = getattr(args, 'monitor_metric', 'val_loss')
        if monitor_metric == 'val_auc':
            es_mode = 'max'
        else:
            es_mode = 'min'
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True,
                                       mode=es_mode, monitor_metric=monitor_metric)
        early_stopping_loss = EarlyStopping(patience=20, stop_epoch=50, verbose=True,
                                           mode='min', monitor_metric='val_loss')
        early_stopping_auc = EarlyStopping(patience=20, stop_epoch=50, verbose=True,
                                           mode='max', monitor_metric='val_auc')
    else:
        early_stopping = None
        early_stopping_loss = None
        early_stopping_auc = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            # CLAM 模型：使用联合 bag+instance 损失训练
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,
                          warmup_epochs=getattr(args, 'warmup_bag_only_epochs', 0),
                          attention_entropy_weight=getattr(args, 'attention_entropy_weight', 0.0))
            val_loss, val_auc, stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

            # Track both loss and AUC best models if dual checkpoint saving enabled
            if getattr(args, 'save_best_auc_ckpt', False) and early_stopping_loss is not None:
                early_stopping_loss(epoch, val_loss, model,
                                  ckpt_name=os.path.join(args.results_dir, "s_{}_checkpoint_loss.pt".format(cur)),
                                  val_auc=val_auc)
                early_stopping_auc(epoch, val_loss, model,
                                  ckpt_name=os.path.join(args.results_dir, "s_{}_checkpoint_auc.pt".format(cur)),
                                  val_auc=val_auc)

        else:
            # 基础 MIL 模型：仅使用 bag 级别损失
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            val_loss, val_auc, stop = validate(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

            # Track both loss and AUC best models if dual checkpoint saving enabled
            if getattr(args, 'save_best_auc_ckpt', False) and early_stopping_loss is not None:
                early_stopping_loss(epoch, val_loss, model,
                                  ckpt_name=os.path.join(args.results_dir, "s_{}_checkpoint_loss.pt".format(cur)),
                                  val_auc=val_auc)
                early_stopping_auc(epoch, val_loss, model,
                                  ckpt_name=os.path.join(args.results_dir, "s_{}_checkpoint_auc.pt".format(cur)),
                                  val_auc=val_auc)

        if stop:
            break

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('train/lr', current_lr, epoch)

    if args.early_stopping:
        # 加载验证集最优 checkpoint
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        # 直接保存最后一个 epoch 的模型
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # ========== SWA: Stochastic Weight Averaging ==========
    use_swa = getattr(args, 'use_swa', False)
    swa_start_epoch = getattr(args, 'swa_start_epoch', 10)
    swa_lr = getattr(args, 'swa_lr', 1e-5)

    if use_swa and epoch >= swa_start_epoch:
        print(f'\n=== SWA enabled: start_epoch={swa_start_epoch}, swa_lr={swa_lr} ===')
        try:
            from torch.optim.swa_utils import AveragedModel, SWALR

            # 创建 SWA 模型
            swa_model = AveragedModel(model)
            swa_model.update_parameters(model)

            # 用较低学习率对最终模型做微调（1 epoch）
            swa_optimizer = torch.optim.Adam(model.parameters(), lr=swa_lr)
            swa_scheduler = SWALR(swa_optimizer, swa_lr=swa_lr)

            # 在训练数据上做 1 个 epoch 的 SWA 更新
            model.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device)
                swa_optimizer.zero_grad()
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
                loss = loss_fn(logits, label)
                loss.backward()
                swa_optimizer.step()
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # 用 SWA 平均后的模型替换
            model.load_state_dict(swa_model.module.state_dict())
            print('SWA completed. Model updated with averaged weights.')
        except Exception as e:
            print(f'SWA skipped due to error: {e}')
    # =====================================================

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,
                   warmup_epochs=0, attention_entropy_weight=0.0):
    """
    CLAM 专用训练循环，同时优化 bag-level loss 和 instance-level clustering loss。
    总损失 = bag_weight * bag_loss + (1 - bag_weight) * instance_loss

    warmup_epochs: 前 N 个 epoch 只用 bag loss，不启用 instance clustering
    attention_entropy_weight: attention 熵正则权重，防止 attention 过于尖锐
    """
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # Warmup 期间禁用 instance loss
    use_instance_loss = epoch >= warmup_epochs

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # 前向传播，启用 instance_eval 以获取实例级损失
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        # 只在非 warmup 期间计算 instance loss
        if use_instance_loss:
            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            # 加权组合 bag loss 和 instance loss
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
        else:
            total_loss = loss

        # Attention entropy regularization
        if attention_entropy_weight > 0:
            A = instance_dict.get('attention', None)
            if A is not None:
                eps = 1e-8
                # 计算 attention 分布的熵（越均匀熵越高）
                entropy = -(A * torch.log(A + eps)).sum(dim=1).mean()
                # 最大化熵 = 最小化负熵
                total_loss = total_loss - attention_entropy_weight * entropy

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            if use_instance_loss:
                print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) +
                    'label: {}, bag_size: {}'.format(label.item(), data.size(0)))
            else:
                print('batch {}, loss: {:.4f}, (inst_disabled), label: {}, bag_size: {}'.format(
                    batch_idx, loss_value, label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if use_instance_loss:
        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss: (disabled), train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    """通用 MIL 训练循环，仅使用 bag-level 分类损失。"""
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    """
    通用验证循环（对应 train_loop 使用的基础 MIL 模型）。
    计算验证集损失、错误率和 AUC，并决定是否触发早停。
    """
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    # 二分类取正类概率列，多分类使用 OvR 策略
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    # Return val_loss, val_auc, and early_stop flag
    if early_stopping:
        assert results_dir
        # Pass both loss and AUC to early_stopping
        early_stopping(epoch, val_loss, model,
                      ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
                      val_auc=auc)

        if early_stopping.early_stop:
            print("Early stopping")
            return val_loss, auc, True

    return val_loss, auc, False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    """
    CLAM 专用验证循环（对应 train_loop_clam）。
    在标准验证基础上额外计算 instance-level 聚类损失，帮助监控实例级学习质量。
    多分类 AUC 使用逐类 ROC 曲线计算后取 nanmean。
    """
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        # 多分类：逐类计算 AUC，对缺失类别填 NaN，最终取 nanmean
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    # Return val_loss, val_auc, and early_stop flag
    if early_stopping:
        assert results_dir
        # Pass both loss and AUC to early_stopping
        early_stopping(epoch, val_loss, model,
                      ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
                      val_auc=auc)

        if early_stopping.early_stop:
            print("Early stopping")
            return val_loss, auc, True

    return val_loss, auc, False

def summary(model, loader, n_classes):
    """
    在给定数据集上执行完整推理，返回 patient-level 结果字典、
    测试错误率、AUC 和 Accuracy_Logger。
    结果字典格式：{slide_id: {'slide_id', 'prob', 'label'}}
    """
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
