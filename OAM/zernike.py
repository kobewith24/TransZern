"""
Standalone Zernike polynomial generation for OAM spiral-spectrum analysis.
This file is intentionally self-contained and does not import from other
folders in the Aberration_correction project.
"""

import math
import torch


def zernike_radial(n, m, rho):
    """Radial part of the Zernike polynomial."""
    R = torch.zeros_like(rho)
    if (n - m) % 2 == 0:
        k_max = (n - m) // 2
        for k in range(k_max + 1):
            term1 = (n + m) // 2 - k
            term2 = (n - m) // 2 - k
            if term1 >= 0 and term2 >= 0:
                numerator = (-1) ** k * math.factorial(n - k)
                denominator = (
                    math.factorial(k)
                    * math.factorial(term1)
                    * math.factorial(term2)
                )
                R = R + numerator / denominator * rho ** (n - 2 * k)
    return R


def zernike_polynomial(n, m, rho, theta):
    """Zernike polynomial Z_n^m in polar coordinates."""
    if m >= 0:
        Z = zernike_radial(n, m, rho) * torch.cos(m * theta)
    else:
        Z = zernike_radial(n, -m, rho) * torch.sin(-m * theta)
    return Z


def generate_phase_map(coefficients, grid_size=400, device='cpu'):
    """
    Build a phase map from a vector of Zernike coefficients.
    Coefficients are assumed to be indexed sequentially in the
    (n, m) enumeration: (0,0), (1,-1), (1,1), (2,-2), (2,0), (2,2), ...
    """
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X ** 2 + Y ** 2)
    Theta = torch.atan2(Y, X)

    # Enforce circular aperture
    R = torch.where(R <= 1.0, R, torch.ones_like(R))

    phase_map = torch.zeros_like(R)
    index = 0
    n = 0
    while index < len(coefficients):
        for m in range(-n, n + 1, 2):
            if index < len(coefficients):
                phase_map = phase_map + coefficients[index] * zernike_polynomial(n, m, R, Theta)
                index += 1
        n += 1
    return phase_map


def zernike_phase_from_coeffs(coeffs, grid_size=400, device='cpu'):
    """Convert a coefficient vector to a phase map (in radians)."""
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.to(device).float()
    else:
        coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    return generate_phase_map(coeffs, grid_size=grid_size, device=device)


def generate_random_coefficients(num_modes=15, device='cpu', symmetric=True):
    """
    Generate random Zernike coefficients.
    The first coefficient (piston, Z_1) is always zero.
    If symmetric=True, ranges are centered at zero to match the
    revised manuscript; otherwise positive-only ranges are used.
    """
    coeffs = torch.zeros(num_modes, device=device)
    if symmetric:
        coeffs[1:3] = torch.empty(2, device=device).uniform_(-15, 15)    # Z2, Z3 tilt
        coeffs[3:6] = torch.empty(3, device=device).uniform_(-12, 12)    # Z4-Z6
        coeffs[6:10] = torch.empty(4, device=device).uniform_(-9, 9)     # Z7-Z10
        coeffs[10:15] = torch.empty(5, device=device).uniform_(-6, 6)    # Z11-Z15
    else:
        coeffs[1:3] = torch.empty(2, device=device).uniform_(0, 15)
        coeffs[3:6] = torch.empty(3, device=device).uniform_(0, 12)
        coeffs[6:10] = torch.empty(4, device=device).uniform_(0, 9)
        coeffs[10:15] = torch.empty(5, device=device).uniform_(0, 6)
    return coeffs


if __name__ == "__main__":
    # Quick sanity check
    coeffs = generate_random_coefficients()
    phase = zernike_phase_from_coeffs(coeffs, grid_size=400)
    print("Phase map shape:", phase.shape)
    print("Coefficient range:", coeffs.min().item(), coeffs.max().item())
