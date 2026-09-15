"""
Optical field propagation and LG vortex beam generation for OAM analysis.
Self-contained; does not import from other project folders.
"""

import math
import torch


def angular_spectrum_propagation(U_in, wavelength=532e-9, z=300e-3, L=6e-3, device='cpu'):
    """
    Propagate a complex optical field using the angular spectrum method.

    Parameters
    ----------
    U_in : torch.Tensor
        Input complex field (N, M).
    wavelength : float
        Wavelength in meters.
    z : float
        Propagation distance in meters.
    L : float
        Physical size of the computational window in meters.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    U_out : torch.Tensor
        Propagated complex field.
    """
    N, M = U_in.shape
    dx = L / N
    dy = dx
    k = 2 * math.pi / wavelength

    fx = torch.fft.fftshift(torch.fft.fftfreq(M, d=dx, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(N, d=dy, device=device))
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')

    # Evanescent waves are set to zero
    evanescent_mask = (wavelength ** 2) * (FX ** 2 + FY ** 2) >= 1.0
    H = torch.exp(1j * k * z * torch.sqrt(1.0 - (wavelength ** 2) * (FX ** 2 + FY ** 2)))
    H[evanescent_mask] = 0.0

    Uf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))
    Uf_prop = Uf * H
    U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Uf_prop)))
    return U_out


def generate_lg_vortex(grid_size=400, l=20, L=6e-3, w=2e-3, device='cpu', apply_aperture=True):
    """
    Generate a Laguerre-Gaussian vortex beam waist-plane field.

    Parameters
    ----------
    grid_size : int
        Number of pixels per side.
    l : int
        Topological charge.
    L : float
        Physical window size in meters.
    w : float
        Beam waist radius in meters.
    device : str
        'cpu' or 'cuda'.
    apply_aperture : bool
        Whether to apply a soft circular aperture matching the SLM active area.

    Returns
    -------
    E : torch.Tensor
        Complex field (grid_size, grid_size).
    """
    dx = L / grid_size
    x = dx * (torch.arange(grid_size, device=device) - grid_size // 2)
    y = dx * (torch.arange(grid_size, device=device) - grid_size // 2)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Helical phase
    phase = l * torch.atan2(Y, X)

    # Gaussian envelope
    gauss_envelope = torch.exp(-(X ** 2 + Y ** 2) / w ** 2)

    E = gauss_envelope * torch.exp(1j * phase)

    if apply_aperture:
        # Soft circular aperture with radius ~ 90% of the half-window
        aperture_radius = L * 0.45
        aperture = torch.exp(-(X ** 2 + Y ** 2) / aperture_radius ** 2)
        E = E * aperture

    return E


def simulate_aberrated_field(coeffs, l=20, wavelength=532e-9, z=300e-3,
                             L=6e-3, w=2e-3, grid_size=400, device='cpu'):
    """
    Simulate an aberrated LG beam intensity pattern at distance z.

    Parameters
    ----------
    coeffs : torch.Tensor
        15-element Zernike coefficient vector (radians), piston first.
    l : int
        Topological charge.
    wavelength, z, L, w : float
        Optical parameters.
    grid_size : int
        Simulation grid size.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    E_out : torch.Tensor
        Propagated complex field.
    """
    from zernike import zernike_phase_from_coeffs

    E0 = generate_lg_vortex(grid_size=grid_size, l=l, L=L, w=w, device=device)
    phi_aber = zernike_phase_from_coeffs(coeffs, grid_size=grid_size, device=device)
    E_aber = E0 * torch.exp(1j * phi_aber)
    E_out = angular_spectrum_propagation(E_aber, wavelength=wavelength, z=z, L=L, device=device)
    return E_out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    E = generate_lg_vortex(grid_size=400, l=20, device=device)
    E_prop = angular_spectrum_propagation(E, device=device)
    print("Input shape:", E.shape)
    print("Output shape:", E_prop.shape)
    print("Output intensity range:", E_prop.abs().pow(2).min().item(), E_prop.abs().pow(2).max().item())
