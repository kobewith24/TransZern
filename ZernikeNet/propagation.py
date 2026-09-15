import torch
import math


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