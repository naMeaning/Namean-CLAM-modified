# =============================================================================
# 文件功能：通用工具函数模块
# 提供以下功能：
#   - SubsetSequentialSampler : 按给定索引顺序采样（不重复）
#   - collate_MIL             : MIL 数据批次整理（bag 特征 + 标签）
#   - collate_features        : 特征提取批次整理（特征 + 坐标）
#   - get_simple_loader       : 构建顺序 DataLoader
#   - get_split_loader        : 按模式（训练/测试/加权）构建 DataLoader
#   - get_optim               : Adam / SGD 优化器工厂
#   - print_network           : 打印网络参数量
#   - generate_split          : K-Fold 数据集划分生成器
#   - nth                     : 从迭代器中取第 n 个元素
#   - calculate_error         : 计算批次错误率
#   - make_weights_for_balanced_classes_split : 类别均衡权重
#   - initialize_weights      : Xavier 权重初始化
# =============================================================================

import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		# 按给定索引顺序依次返回，保证测试时顺序可复现
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	# 将同一 bag 内所有 patch 特征沿 batch 维度拼接
	# 标签统一封装为 LongTensor
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_features(batch):
	# 特征提取时使用：拼接图像特征并垂直堆叠坐标数组
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	# 构建顺序 DataLoader，用于评估阶段（固定顺序，无随机性）
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
	根据当前模式返回合适的 DataLoader：
	  - testing=True  : 随机子集采样（仅取 10%，用于快速调试）
	  - training=True, weighted=True : 加权随机采样（解决类别不平衡）
	  - training=True, weighted=False: 完全随机采样
	  - 其他（验证/测试）: 顺序采样
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		# 调试模式：随机选取 10% 数据，使用顺序采样器
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	# 只对 requires_grad=True 的参数进行优化（冻结参数不更新）
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	# 打印网络结构和参数量（总参数 + 可训练参数）
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	"""
	K-Fold 数据集划分生成器（yield 版本，节省内存）。
	每次 yield 返回 (train_ids, val_ids, test_ids) 三元组。

	策略：
	  1. 若提供 custom_test_ids，则从候选集中排除这些 ID
	  2. 按类别采样 val_ids 和 test_ids
	  3. 剩余样本按 label_frac 比例采样为训练集（半监督场景）
	"""
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				# 半监督：只使用 label_frac 比例的训练样本
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	# 取迭代器的第 n 个元素（跳过前 n 个），用于跳过特定 Fold
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	# 计算批次错误率 = 1 - 准确率
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	# 为类别不平衡数据集计算样本权重，使每个类别被采样概率相等
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	# 对线性层使用 Xavier 正态初始化，对 BatchNorm1d 使用常数初始化
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

