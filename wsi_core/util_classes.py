# =============================================================================
# 文件功能：WSI 处理辅助类
# 包含两个模块：
#   1. Mosaic_Canvas   : 用于将多个 patch 缩放后拼接成网格预览图（Mosaic）
#   2. 轮廓检测函数族   : 判断给定坐标是否位于组织轮廓内部
#      - Contour_Checking_fn : 抽象基类（定义 __call__ 接口）
#      - isInContourV1       : 仅检测 patch 左上角顶点（最快，最宽松）
#      - isInContourV2       : 检测 patch 中心点（默认）
#      - isInContourV3_Easy  : 检测 4 个围绕中心点的采样点，1/4 在内即通过（宽松）
#      - isInContourV3_Hard  : 检测 4 个采样点，全部在内才通过（严格）
# =============================================================================

import os
import numpy as np
from PIL import Image
import pdb
import cv2

class Mosaic_Canvas(object):
	"""
	Mosaic 拼接画布类。
	将多个 patch 等比缩放后，按从左到右、从上到下的网格顺序拼接为预览图。
	
	Args:
		patch_size    : 原始 patch 的一边像素大小（正方形）
		n             : 预计总 patch 数（决定行列布局）
		downscale     : 缩放倍数，实际画布中每个 patch 为 patch_size // downscale
		n_per_row     : 每行显示的 patch 数
		bg_color      : 背景颜色（RGB 元组）
		alpha         : < 0 时使用 RGB 模式，>= 0 时使用 RGBA 并设置透明度
	"""
	def __init__(self,patch_size=256, n=100, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1):
		self.patch_size = patch_size
		self.downscaled_patch_size = int(np.ceil(patch_size/downscale))
		self.n_rows = int(np.ceil(n / n_per_row))
		self.n_cols = n_per_row
		w = self.n_cols * self.downscaled_patch_size
		h = self.n_rows * self.downscaled_patch_size
		if alpha < 0:
			canvas = Image.new(size=(w,h), mode="RGB", color=bg_color)
		else:
			canvas = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
		
		self.canvas = canvas
		self.dimensions = np.array([w, h])
		self.reset_coord()

	def reset_coord(self):
		"""重置当前写入光标位置为画布左上角"""
		self.coord = np.array([0, 0])

	def increment_coord(self):
		"""将写入光标移动到下一个 patch 位置（先向右，到行末则换行）"""
		#print('current coord: {} x {} / {} x {}'.format(self.coord[0], self.coord[1], self.dimensions[0], self.dimensions[1]))
		assert np.all(self.coord<=self.dimensions)
		if self.coord[0] + self.downscaled_patch_size <=self.dimensions[0] - self.downscaled_patch_size:
			self.coord[0]+=self.downscaled_patch_size
		else:
			self.coord[0] = 0 
			self.coord[1]+=self.downscaled_patch_size
		

	def save(self, save_path, **kwargs):
		"""将画布保存为图像文件"""
		self.canvas.save(save_path, **kwargs)

	def paste_patch(self, patch):
		"""将 patch 缩放后粘贴到当前光标位置，并自动移动光标"""
		assert patch.size[0] == self.patch_size
		assert patch.size[1] == self.patch_size
		self.canvas.paste(patch.resize(tuple([self.downscaled_patch_size, self.downscaled_patch_size])), tuple(self.coord))
		self.increment_coord()

	def get_painting(self):
		"""获取当前 canvas PIL Image 对象"""
		return self.canvas

class Contour_Checking_fn(object):
	"""
	轮廓内部检测函数的抽象基类。
	子类必须实现 __call__(pt) 方法，pt 为 (x, y) 坐标，返回 1（在内）或 0（在外）。
	"""
	# Defining __call__ method 
	def __call__(self, pt): 
		raise NotImplementedError

class isInContourV1(Contour_Checking_fn):
	"""
	V1：仅检测 patch 左上角顶点是否在轮廓内。
	速度最快，但对于边界 patch 可能误判，适合快速粗筛。
	"""
	def __init__(self, contour):
		self.cont = contour

	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False) >= 0 else 0

class isInContourV2(Contour_Checking_fn):
	"""
	V2：检测 patch 中心点是否在轮廓内（相比 V1 更准确）。
	center = (pt[0] + patch_size//2, pt[1] + patch_size//2)
	"""
	def __init__(self, contour, patch_size):
		self.cont = contour
		self.patch_size = patch_size

	def __call__(self, pt): 
		pt = np.array((pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)).astype(float)
		return 1 if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False) >= 0 else 0

# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
	"""
	V3 宽松版：检测围绕 patch 中心点的 4 个采样点，任意 1 个在轮廓内即通过。
	center_shift 控制 4 点到中心的偏移距离（相对 patch_size//2 的比例）。
	适用于需要高召回率的场景（尽量不漏掉组织区域的 patch）。
	"""
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0

# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(Contour_Checking_fn):
	"""
	V3 严格版：检测围绕 patch 中心点的 4 个采样点，全部在轮廓内才通过。
	适用于需要高精度的场景（确保选取的 patch 完全位于组织内部，
	避免边界 patch 引入背景噪声）。
	"""
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			# 任何一个点在轮廓外则直接返回 0（失败）
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) < 0:
				return 0
		return 1



		