# =============================================================================
# 文件功能：WSI 核心辅助工具函数库
# 包含以下功能：
#   - 白/黑 patch 判断（精确版和快速版）
#   - Patch 坐标生成器（滑动窗口）
#   - HDF5 文件创建/追加写入（初始化和增量保存）
#   - 注意力分数采样工具（top-k、范围采样、百分位）
#   - 坐标筛选（ROI 区域过滤）
#   - Patch 拼接（从像素和坐标两种方式重建预览图）
#   - 随机采样 patches 工具
# =============================================================================

import h5py
import numpy as np
import os
import pdb
from wsi_core.util_classes import Mosaic_Canvas
from PIL import Image
import math
import cv2
from tqdm import tqdm

def isWhitePatch(patch, satThresh=5):
    """通过 HSV 饱和度通道均值判断 patch 是否为白色/空白（低饱和度 = 白底）"""
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    """通过 RGB 三通道均值判断 patch 是否为黑色（三通道均值均低于阈值）"""
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    """严格版黑色 patch 判断：超过 percentage 比例的像素三通道均 < rgbThresh"""
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    """严格版白色 patch 判断：超过 percentage 比例的像素三通道均 > rgbThresh"""
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def coord_generator(x_start, x_end, x_step, y_start, y_end, y_step, args_dict=None):
    """
    生成二维网格坐标（x, y）的生成器，用于滑动窗口 patch 坐标枚举。
    若 args_dict 不为空，则将坐标嵌入参数字典中 yield，否则直接 yield (x, y) 元组。
    """
    for x in range(x_start, x_end, x_step):
        for y in range(y_start, y_end, y_step):
            if args_dict is not None:
                process_dict = args_dict.copy()
                process_dict.update({'pt':(x,y)})
                yield process_dict
            else:
                yield (x,y)

def savePatchIter_bag_hdf5(patch):
    """将单个 patch 图像追加写入已存在的 HDF5 bag 文件（配合 _getPatchGenerator 使用）。"""
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path= tuple(patch.values())
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x,y)

    file.close()

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """
    通用 HDF5 写入函数，支持创建（mode='w'）和追加（mode='a'）两种模式。
    对 asset_dict 中的每个 key：
      - 若 dataset 不存在，则创建可变长（maxshape=None）dataset 并写入
      - 若已存在，则 resize 并追加新数据
    attr_dict 用于为 dataset 添加元数据属性（如 patch_size、patch_level 等）。
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def initialize_hdf5_bag(first_patch, save_coord=False):
    """
    使用第一个 patch 初始化 HDF5 bag 文件，创建可扩展的 'imgs' dataset。
    若 save_coord=True，同时创建 'coords' dataset 保存坐标信息。
    返回创建的 HDF5 文件路径。
    """
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())
    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs', 
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x,y)

    file.close()
    return file_path

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    """
    从 scores 中采样处于 [start, end] 区间内的 k 个索引。
    convert_to_percentile=True 时将 start/end 解释为百分位数（而非绝对值）。
    用于热图分析中选取"中等"注意力分数的 patch 进行可视化。
    """
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1 
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def top_k(scores, k, invert=False):
    """
    返回 scores 中最高（invert=False）或最低（invert=True）的 k 个索引。
    用于选取高/低注意力 patch 进行可视化分析。
    """
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def to_percentiles(scores):
    """将任意分数数组转换为 [0, 100] 百分位（使用排名/总数 * 100）"""
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    """过滤掉不在指定 ROI（top_left ~ bot_right）矩形区域内的坐标和对应分数"""
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):
    """
    从注意力分数中采样感兴趣区域（ROI）的 patch 坐标。
    支持三种采样模式：
      - 'range_sample' : 采样处于 [score_start, score_end] 百分位区间的 patch
      - 'topk'         : 采样注意力最高的 k 个 patch
      - 'reverse_topk' : 采样注意力最低的 k 个 patch
    """
    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    """在 patch 位置绘制矩形网格线，用于可视化 patch 边界"""
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    """
    从 HDF5 patch 数据集中读取图像，拼接到 canvas 对应坐标位置（基于像素数组）。
    用于从预先保存的 patch 图像重建 WSI 预览图（StitchPatches 内部使用）。
    """
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=True):
    """
    从 WSI 对象在线读取 patch 并拼接到 canvas（基于坐标）。
    无需预先保存 patch 图像，相比 DrawMap 更节省磁盘（StitchCoords 内部使用）。
    """
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
        
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    
    for idx in tqdm(range(total)):        
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    """
    从 HDF5 文件（含 patch 像素数据）重建 WSI 缩略预览图。
    适用于旧版 createPatches_bag_hdf5 生成的含图像数据的 h5 文件。
    """
    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['imgs']
        coords = file['coords'][:]
        if 'downsampled_level_dim' in dset.attrs.keys():
            w, h = dset.attrs['downsampled_level_dim']
        else:
            w, h = dset.attrs['level_dim']

    print('original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print(f'number of patches: {len(coords)}')
    img_shape = dset[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)
    
    return heatmap

def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    """
    从 HDF5 文件（仅含坐标）在线读取 WSI patch 并重建预览图。
    适用于 process_contour 生成的仅含坐标的新版 h5 文件（更省磁盘）。
    """
    wsi = wsi_object.getOpenSlide()
    w, h = wsi.level_dimensions[0]
    print('original size: {} x {}'.format(w, h))
    
    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print('downscaled size for stiching: {} x {}'.format(w, h))

    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        print('start stitching {}'.format(dset.attrs['name']))
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']
    
    print(f'number of patches: {len(coords)}')
    print(f'patch size: {patch_size} x {patch_size} patch level: {patch_level}')
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print(f'ref patch size: {patch_size} x {patch_size}')

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)
    return heatmap

def SamplePatches(coords_file_path, save_file_path, wsi_object, 
    patch_level=0, custom_downsample=1, patch_size=256, sample_num=100, seed=1, stitch=True, verbose=1, mode='w'):
    """
    从坐标文件中随机采样 sample_num 个 patch，在线从 WSI 读取并保存到 HDF5 文件。
    若 stitch=True，同时生成 Mosaic_Canvas 拼接预览图。
    用于可视化检查已分割区域中的代表性 patch 样本。
    """
    with h5py.File(coords_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        h5_patch_size = dset.attrs['patch_size']
        h5_patch_level = dset.attrs['patch_level']
    
    if verbose>0:
        print('in .h5 file: total number of patches: {}'.format(len(coords)))
        print('in .h5 file: patch size: {}x{} patch level: {}'.format(h5_patch_size, h5_patch_size, h5_patch_level))

    if patch_level < 0:
        patch_level = h5_patch_level

    if patch_size < 0:
        patch_size = h5_patch_size

    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(coords)), min(len(coords), sample_num), replace=False)

    target_patch_size = np.array([patch_size, patch_size])
    
    if custom_downsample > 1:
        target_patch_size = (np.array([patch_size, patch_size]) / custom_downsample).astype(np.int32)
        
    if stitch:
        canvas = Mosaic_Canvas(patch_size=target_patch_size[0], n=sample_num, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1)
    else:
        canvas = None
    
    for idx in indices:
        coord = coords[idx]
        patch = wsi_object.wsi.read_region(coord, patch_level, tuple([patch_size, patch_size])).convert('RGB')
        if custom_downsample > 1:
            patch = patch.resize(tuple(target_patch_size))

        # if isBlackPatch_S(patch, rgbThresh=20, percentage=0.05) or isWhitePatch_S(patch, rgbThresh=220, percentage=0.25):
        #     continue

        if stitch:
            canvas.paste_patch(patch)

        asset_dict = {'imgs': np.array(patch)[np.newaxis,...], 'coords': coord}
        save_hdf5(save_file_path, asset_dict, mode=mode)
        mode='a'

    return canvas, len(coords), len(indices)