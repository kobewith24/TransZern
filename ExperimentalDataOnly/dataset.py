import torch
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, data_dir="Joint/datasets", device='cpu'):
        self.data_dir = data_dir
        self.device = device

        self.real_dir = os.path.join(data_dir, "real_intensity")
        self.sim_dir = os.path.join(data_dir, "sim_intensity")
        self.csv_dir = os.path.join(data_dir, "csv")

        self.real_files = sorted([f for f in os.listdir(self.real_dir) if f.endswith('.png')])
        self.sim_files = sorted([f for f in os.listdir(self.sim_dir) if f.endswith('.png')])
        self.csv_files = sorted([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')])

        min_len = min(len(self.real_files), len(self.sim_files), len(self.csv_files))
        self.real_files = self.real_files[:min_len]
        self.sim_files = self.sim_files[:min_len]
        self.csv_files = self.csv_files[:min_len]
    
    def _extract_number(self, filename):
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0
    
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
    
    def _normalize_brightness(self, real_tensor, sim_tensor):
        real_brightness = torch.mean(real_tensor)
        sim_brightness = torch.mean(sim_tensor)
        
        if real_brightness > 1e-6:
            brightness_ratio = sim_brightness / real_brightness
            normalized_real = real_tensor * brightness_ratio
            normalized_real = torch.clamp(normalized_real, 0, 1)
            return normalized_real
        else:
            return real_tensor
    
    def __len__(self):
        return len(self.real_files)
    
    def __getitem__(self, idx):
        # 加载实采光场图
        real_path = os.path.join(self.real_dir, self.real_files[idx])
        real_img = self._load_image(real_path)
        # 加载仿真光场图
        sim_path = os.path.join(self.sim_dir, self.sim_files[idx])
        sim_img = self._load_image(sim_path)
        # 加载Zernike系数
        csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
        coeffs = self._load_coefficients(csv_path)

        real_img = self._normalize_brightness(real_img, sim_img)
        
        return real_img, sim_img, coeffs