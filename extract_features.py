# =============================================================================
# 文件功能：特征提取脚本
# 流程：
#   1. 读取 CSV 文件获取 bag（.h5 格式 patch 文件）列表
#   2. 使用预训练编码器（ResNet-50 截断 / UNI / CONCH）对每个 patch 提取特征
#   3. 将特征和坐标保存为 .h5 文件，再转换为 .pt 格式（无坐标，供 MIL 训练使用）
# 支持的编码器：resnet50_trunc, uni_v1, conch_v1
# =============================================================================

import time
import os
import argparse
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide

from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag, get_eval_transforms
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	使用给定 DataLoader 和编码器模型批量提取 patch 特征，
	以追加写入（mode='a'）方式将每批结果保存到 .h5 文件。
	
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'  # 第一个 batch 以写模式创建文件
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			# 编码器前向推理，得到 patch 的嵌入特征向量
			features = model(batch)
			
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'  # 后续 batch 以追加模式写入同一文件
	
	return output_path


# =============================================================================
# 命令行参数解析
# =============================================================================
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str)          # .h5 patch 文件所在目录
parser.add_argument('--csv_path', type=str)           # 包含 slide 列表的 CSV 文件
parser.add_argument('--feat_dir', type=str)           # 特征输出目录
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')  # 禁用自动跳过已处理 slide
parser.add_argument('--target_patch_size', type=int, default=224,
					help='the desired size of patches for scaling before feature embedding')
args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	bags_dataset = Dataset_All_Bags(csv_path)
	
	# 创建输出目录（h5_files 和 pt_files 子目录）
	os.makedirs(args.feat_dir, exist_ok=True)
	dest_files = os.listdir(args.feat_dir)

	# 加载编码器模型及对应的图像预处理 transform
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)		
	model = model.to(device)
	_ = model.eval()

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
	
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id + '.h5'
		bag_candidate = os.path.join(args.data_dir, 'patches', bag_name)

		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(bag_name)
		# 自动跳过已提取特征的 slide（检查 pt_files 目录中是否已存在）
		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		file_path = bag_candidate
		time_start = time.time()

		# 构建 patch 数据集和 DataLoader
		dataset = Whole_Slide_Bag(file_path=file_path, img_transforms=img_transforms)
		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		# 打印特征维度以便验证
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		# 将特征从 h5 转换为 PyTorch .pt 格式（去除坐标，供 MIL 训练直接使用）
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
