import torch
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset


class DirectDataset(Dataset):
    """消融实验数据集: 直接使用原始实验光强训练 ZernikeNet (无 ConvertNet).
    
    特点:
    - 不加载任何仿真数据
    - 不做亮度归一化（因为亮度归一化依赖仿真参考图）
    - 完全在原始实验域上训练和测试
    """
    def __init__(self, data_dir="SimulatedDataOnly/datasets", device='cpu'):
        self.data_dir = data_dir
        self.device = device

        self.real_dir = os.path.join(data_dir, "real_intensity")
        self.csv_dir = os.path.join(data_dir, "csv")

        self.real_files = sorted([f for f in os.listdir(self.real_dir) if f.endswith('.png')])
        self.csv_files = sorted([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')])

        min_len = min(len(self.real_files), len(self.csv_files))
        self.real_files = self.real_files[:min_len]
        self.csv_files = self.csv_files[:min_len]
    
    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img.astype('float32') / 255.0).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def _load_coefficients(self, path):
        df = pd.read_csv(path, header=None)
        coeffs = torch.from_numpy(df.values.flatten().astype('float32'))
        if len(coeffs) < 15:
            padded_coeffs = torch.zeros(15, device=self.device)
            padded_coeffs[:len(coeffs)] = coeffs
            coeffs = padded_coeffs
        elif len(coeffs) > 15:
            coeffs = coeffs[:15]
        return coeffs.to(self.device)
    
    def __len__(self):
        return len(self.real_files)
    
    def __getitem__(self, idx):
        # 只加载原始实验光场图，不做任何仿真化处理
        real_path = os.path.join(self.real_dir, self.real_files[idx])
        real_img = self._load_image(real_path)
        
        csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
        coeffs = self._load_coefficients(csv_path)
        
        return real_img, coeffs
