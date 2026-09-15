import torch
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# ============ 涡旋光束生成 ============
def generate_vortex_beam(grid_size=400, l=20, L=6e-3, w=2e-3, device='cpu'):
    dx = L / grid_size
    dy = dx
    x = dx * (torch.arange(grid_size, device=device) - grid_size // 2)
    y = dy * (torch.arange(grid_size, device=device) - grid_size // 2)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    phase = l * torch.atan2(Y, X)
    
    gauss_envelope = torch.exp(-(X**2 + Y**2) / w**2)
    aperture_radius = L * 0.4
    aperture = torch.exp(-(X**2 + Y**2) / aperture_radius**2)
    E = gauss_envelope * aperture * torch.exp(1j * phase)

    return E

# ============ 角谱传播 ============
def angular_spectrum_propagation(U_in, wavelength=532e-9, z=300e-3, L=6e-3, device='cpu'):
    N, M = U_in.shape
    dx = L / N
    dy = dx
    k = 2 * math.pi / wavelength

    fx = torch.fft.fftshift(torch.fft.fftfreq(M, d=dx, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(N, d=dy, device=device))
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')

    H = torch.exp(1j * k * z * torch.sqrt(1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)))
    H[(wavelength ** 2) * (FX ** 2 + FY ** 2) >= 1] = 0

    Uf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))
    Uf_prop = Uf * H
    U_out = torch.fft.fftshift(torch.fft.ifft2(Uf_prop))
    
    return U_out

# ============ Zernike相关函数 ============
def zernike_radial(n, m, rho):
    R = torch.zeros_like(rho)
    if (n - m) % 2 == 0:
        k_max = (n - m) // 2
        for k in range(k_max + 1):
            term1 = (n + m) // 2 - k
            term2 = (n - m) // 2 - k
            if term1 >= 0 and term2 >= 0:
                numerator = (-1)**k * math.factorial(n - k)
                denominator = (
                    math.factorial(k) *
                    math.factorial(term1) *
                    math.factorial(term2)
                )
                R = R + numerator / denominator * rho**(n - 2*k)
    return R

def zernike_polynomial(n, m, rho, theta):
    if m >= 0:
        Z = zernike_radial(n, m, rho) * torch.cos(m * theta)
    else:
        Z = zernike_radial(n, -m, rho) * torch.sin(-m * theta)
    return Z

def generate_phase_map(coefficients, grid_size=2000, device='cpu'):
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2)
    Theta = torch.atan2(Y, X)
    phase_map = torch.zeros_like(R)
    index = 0
    n = 0
    while index < len(coefficients):
        for m in range(-n, n+1, 2):
            if index < len(coefficients):
                phase_map = phase_map + coefficients[index] * zernike_polynomial(n, m, R, Theta)
                index += 1
        n += 1
    phase_map = (phase_map + math.pi) % (2*math.pi)
    return phase_map

def zernike_phase_from_coeffs(coeffs, grid_size=400, device='cpu'):
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.to(device).float()
    else:
        coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    phase = generate_phase_map(coeffs, grid_size=grid_size, device=device)
    return phase

def generate_aberration_phase(num_modes=15, grid_size=400, device='cpu'):
    coeffs = torch.zeros(num_modes, device=device)
    coeffs[0] = 0
    coeffs[1:3] = torch.empty(2, device=device).uniform_(0, 15)
    coeffs[3:6] = torch.empty(3, device=device).uniform_(0, 12)
    coeffs[6:10] = torch.empty(4, device=device).uniform_(0, 9)
    coeffs[10:15] = torch.empty(5, device=device).uniform_(0, 6)
    phase = zernike_phase_from_coeffs(coeffs, grid_size=grid_size, device=device)
    return phase, coeffs


# 物理一致性损失
class PhysicalInformedLoss(torch.nn.Module):
    def __init__(self, coeff_weight=1.0, intensity_weight=0.3, epsilon=1e-8):
        super().__init__()
        self.coeff_weight = coeff_weight
        self.intensity_weight = intensity_weight
        self.epsilon = epsilon
        
    def forward(self, coeffs_pred, coeffs_true, I_distorted, device):
        coeff_loss = torch.nn.MSELoss()(coeffs_pred[:, 1:15], coeffs_true[:, 1:15])
        intensity_loss = self.compute_intensity_consistency_loss(
            coeffs_pred, coeffs_true, I_distorted, device
        )
        total_loss = (self.coeff_weight * coeff_loss + self.intensity_weight * intensity_loss)
        return total_loss
    
    def compute_intensity_consistency_loss(self, coeffs_pred, coeffs_true, I_distorted, device):
        batch_size = coeffs_pred.size(0)
        intensity_losses = []
        grid_size = 400
        l = 20
        L = 6e-3
        w = 2e-3
        wavelength = 532e-9
        z = 300e-3
        
        for i in range(batch_size):
            E0 = generate_vortex_beam(grid_size, l, L, w, device=device)
            phi_aberr_true = zernike_phase_from_coeffs(
                coeffs_true[i], grid_size=grid_size, device=device
            )
            E_aberr = E0 * torch.exp(1j * phi_aberr_true)
            phi_correct_pred = zernike_phase_from_coeffs(
                coeffs_pred[i], grid_size=grid_size, device=device
            )
            E_corrected = E_aberr * torch.exp(-1j * phi_correct_pred)
            U_corrected = angular_spectrum_propagation(
                E_corrected, wavelength=wavelength, z=z, L=L, device=device
            )
            I_corrected_pred = torch.abs(U_corrected) ** 2
            
            U_ideal = angular_spectrum_propagation(
                E0, wavelength=wavelength, z=z, L=L, device=device
            )
            I_ideal = torch.abs(U_ideal) ** 2
            
            I_pred_norm = (I_corrected_pred - I_corrected_pred.min()) / (I_corrected_pred.max() - I_corrected_pred.min() + self.epsilon)
            I_ideal_norm = (I_ideal - I_ideal.min()) / (I_ideal.max() - I_ideal.min() + self.epsilon)
            
            loss = torch.nn.MSELoss()(I_pred_norm, I_ideal_norm)
            intensity_losses.append(loss)
        
        return torch.mean(torch.stack(intensity_losses))


def normalize_intensity(intensity):
    intensity_np = intensity.cpu().numpy()
    intensity_normalized = ((intensity_np - intensity_np.min()) / 
                           (intensity_np.max() - intensity_np.min() + 1e-8) * 255).astype(np.uint8)
    return intensity_normalized

def save_intensity_image(intensity, path):
    intensity_norm = normalize_intensity(intensity)
    cv2.imwrite(path, intensity_norm)

def load_intensity_image(path, device='cpu'):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    tensor = torch.from_numpy(img.astype('float32') / 255.0).unsqueeze(0)
    return tensor.to(device)

def load_coefficients(path, device='cpu'):
    df = pd.read_csv(path, header=None)
    coeffs = torch.from_numpy(df.values.flatten().astype('float32'))
    if len(coeffs) < 15:
        padded_coeffs = torch.zeros(15, device=device)
        padded_coeffs[:len(coeffs)] = coeffs
        coeffs = padded_coeffs
    elif len(coeffs) > 15:
        coeffs = coeffs[:15]
    return coeffs.to(device)


def zernike_regularizer(coeffs, device='cpu'):
    weights = torch.exp(-torch.arange(15, device=device) / 6.0)
    return torch.mean((coeffs ** 2) * weights)