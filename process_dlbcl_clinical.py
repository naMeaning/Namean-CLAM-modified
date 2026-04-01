"""
DLBCL 临床数据处理脚本
功能：
1. 数据清洗与二值化
2. Hans算法推导分子亚型
3. 数据校验
4. 输出CLAM格式CSV + PyTorch Dataset
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ========== 1. 加载数据 ==========
DATA_PATH = 'dataset_csv/morph_clinical_data_cleaned.csv'
df = pd.read_csv(DATA_PATH)
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

# ========== 2. 数据清洗与二值化 ==========
# 提取需要的列
ihc_cols = ['CD10 IHC', 'BCL6 IHC', 'MUM1 IHC']
df_work = df[['patient_id'] + ihc_cols + ['HANS']].copy()

# 标记缺失值 (IHC列有任意NaN则标记)
ihc_missing = df_work[ihc_cols].isnull().any(axis=1)
df_work['has_missing'] = ihc_missing
print(f"\nIHC存在缺失值的行数: {ihc_missing.sum()}")

# 二值化函数: 值==1.0 或 >=30 视为阳性(1), 否则为阴性(0)
def binarize(val):
    if pd.isna(val):
        return np.nan
    return 1 if (val == 1.0 or val >= 30) else 0

for col in ihc_cols:
    df_work[f'{col}_binary'] = df_work[col].apply(binarize)

print("\n二值化结果示例:")
print(df_work[['patient_id', 'CD10 IHC', 'CD10 IHC_binary', 'BCL6 IHC', 'BCL6 IHC_binary', 'MUM1 IHC', 'MUM1 IHC_binary']].head(10))

# ========== 3. Hans算法推导 ==========
def derive_hans(row):
    """Hans算法: 1=GCB, 0=non-GCB"""
    cd10 = row['CD10 IHC_binary']
    bcl6 = row['BCL6 IHC_binary']
    mum1 = row['MUM1 IHC_binary']

    # 任一关键字段缺失则无法推导
    if pd.isna(cd10) or pd.isna(bcl6) or pd.isna(mum1):
        return np.nan

    # Hans算法逻辑
    if cd10 == 1:
        return 1  # CD10阳性 -> GCB
    else:  # CD10阴性
        if bcl6 == 0:
            return 0  # CD10- 且 BCL6- -> non-GCB
        else:  # BCL6阳性
            if mum1 == 0:
                return 1  # CD10- 且 BCL6+ 且 MUM1- -> GCB
            else:
                return 0  # CD10- 且 BCL6+ 且 MUM1+ -> non-GCB

df_work['Derived_Subtype'] = df_work.apply(derive_hans, axis=1)

# ========== 4. 数据校验 ==========
# 筛选有完整Hans推导结果的行
valid_derived = df_work['Derived_Subtype'].notna()
valid_original = df_work['HANS'].notna()
both_valid = valid_derived & valid_original

consistent = (df_work.loc[both_valid, 'Derived_Subtype'] == df_work.loc[both_valid, 'HANS']).sum()
inconsistent = (~(df_work.loc[both_valid, 'Derived_Subtype'] == df_work.loc[both_valid, 'HANS'])).sum()
skipped_missing = len(df_work) - both_valid.sum()

print("\n" + "="*50)
print("数据校验结果:")
print(f"  一致的样本数: {consistent}")
print(f"  不一致的样本数: {inconsistent}")
print(f"  因缺失被跳过的样本数: {skipped_missing}")
print("="*50)

# 打印不一致的样本详情
if inconsistent > 0:
    print("\n不一致样本详情:")
    mismatch = df_work[both_valid & (df_work['Derived_Subtype'] != df_work['HANS'])]
    print(mismatch[['patient_id', 'CD10 IHC_binary', 'BCL6 IHC_binary', 'MUM1 IHC_binary', 'HANS', 'Derived_Subtype']])

# ========== 5. 输出CLAM格式CSV ==========
# 筛选有效数据: 有Derived_Subtype的样本
df_clean = df_work[df_work['Derived_Subtype'].notna()].copy()

# 转换为CLAM格式: case_id, slide_id, label (同时保留二值化特征)
df_cllam = pd.DataFrame({
    'case_id': df_clean['patient_id'].astype(str),
    'slide_id': df_clean['patient_id'].astype(str),
    'label': df_clean['Derived_Subtype'].astype(int).map({1: 'GCB', 0: 'non-GCB'}),
    'CD10_binary': df_clean['CD10 IHC_binary'].astype(int),
    'BCL6_binary': df_clean['BCL6 IHC_binary'].astype(int),
    'MUM1_binary': df_clean['MUM1 IHC_binary'].astype(int),
})

# 输出CLAM格式CSV (仅保留case_id, slide_id, label)
OUTPUT_PATH = 'dataset_csv/dlbcl_morph.csv'
df_cllam[['case_id', 'slide_id', 'label']].to_csv(OUTPUT_PATH, index=False)
print(f"\n输出CLAM格式CSV: {OUTPUT_PATH}")
print(f"总样本数: {len(df_cllam)}")
print(f"GCB: {(df_cllam['label']=='GCB').sum()}, non-GCB: {(df_cllam['label']=='non-GCB').sum()}")

# ========== 6. PyTorch Dataset ==========
class DLBCLClinicalDataset(Dataset):
    """DLBCL临床数据PyTorch Dataset"""
    def __init__(self, csv_path, feature_cols=None):
        self.df = pd.read_csv(csv_path)
        # 默认使用二值化后的IHC特征
        if feature_cols is None:
            feature_cols = ['CD10_binary', 'BCL6_binary', 'MUM1_binary']
        self.feature_cols = feature_cols
        # 标签映射: GCB=1, non-GCB=0
        self.label_map = {'GCB': 1, 'non-GCB': 0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 特征: 二值化IHC
        features = torch.tensor(
            [row[col] for col in self.feature_cols],
            dtype=torch.float32
        )
        # 标签
        label = self.label_map[row['label']]
        return features, torch.tensor(label, dtype=torch.long)

# 示例: 实例化DataLoader
if __name__ == '__main__':
    # 使用处理好的CSV创建Dataset
    dataset = DLBCLClinicalDataset('dataset_csv/dlbcl_hans_derived.csv')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("\nPyTorch DataLoader 示例:")
    for batch_x, batch_y in dataloader:
        print(f"  Batch特征: {batch_x.shape}, 标签: {batch_y}")
        break