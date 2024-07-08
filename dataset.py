import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集目录路径
        :param transform: 可选的转换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('_noisy.xyz')]
        self.clean_filenames = [fn.replace('_noisy.xyz', '_clean.xyz') for fn in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        noisy_path = self.filenames[idx]
        clean_path = self.clean_filenames[idx]
        noisy_points = np.loadtxt(noisy_path, skiprows=1)
        clean_points = np.loadtxt(clean_path, skiprows=1)
        sample = {'noisy': noisy_points, 'clean': clean_points}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

def to_tensor(sample):
    noisy, clean = sample['noisy'], sample['clean']
    return {'noisy': torch.tensor(noisy, dtype=torch.float32), 'clean': torch.tensor(clean, dtype=torch.float32)}