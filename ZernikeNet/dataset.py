import torch
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset
from lg_vortex import generate_vortex_beam
from zernike import generate_aberration_phase
from propagation import angular_spectrum_propagation

def generate_training_pair(grid_size=400, l=20, num_modes=15, L=6e-3, w=2e-3, device='cpu'):    # noqa: E741
    E0 = generate_vortex_beam(grid_size, l, L, w, device=device)    # 理想涡旋光束
    phi_aberr, coeffs = generate_aberration_phase(num_modes, grid_size, device=device)
    E_aberr = E0 * torch.exp(1j * phi_aberr)    # 叠加像差后的畸变光场
    I_distorted = torch.abs(angular_spectrum_propagation(E_aberr, wavelength=532e-9, z=300e-3, L=L, device=device)) ** 2    # 畸变光强
    I_ideal = torch.abs(angular_spectrum_propagation(E0, wavelength=532e-9, z=300e-3, L=L, device=device)) ** 2    # 理想光强
    return I_distorted, I_ideal, coeffs, phi_aberr, E0, E_aberr

class PreGeneratedDataset(Dataset):
    def __init__(self, dataset_dir="ZernikeNet/datasets", device='cpu'):
        self.dataset_dir = dataset_dir
        self.device = device
        img_dir = os.path.join(dataset_dir, "img_simulated")
        csv_dir = os.path.join(dataset_dir, "csv_simulated")
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        self.csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])

    def _load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img.astype('float32') / 255.0)
        return img_tensor.to(self.device)

    def _load_coefficients(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        coeffs = torch.from_numpy(df.values.flatten().astype('float32'))
        coeffs = coeffs[:15]
        coeffs_normalized = torch.zeros_like(coeffs)
        coeffs_normalized[0] = 0
        coeffs_normalized[1:3] = coeffs[1:3] / 15.0
        coeffs_normalized[3:6] = coeffs[3:6] / 12.0
        coeffs_normalized[6:10] = coeffs[6:10] / 9.0
        coeffs_normalized[10:15] = coeffs[10:15] / 6.0
        return coeffs.to(self.device)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, "img_simulated", self.img_files[idx])
        distorted_img = self._load_image(img_path)
        csv_path = os.path.join(self.dataset_dir, "csv_simulated", self.csv_files[idx])
        coeffs = self._load_coefficients(csv_path)
        input_tensor = distorted_img.unsqueeze(0)    # (1, H, W)
        return input_tensor, coeffs 