"""
OAM spiral spectrum analysis for TransZern-corrected vortex beams.
Self-contained; does not import from other project folders.
"""

import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from zernike import generate_random_coefficients
from optics import generate_lg_vortex, angular_spectrum_propagation


def compute_oam_spectrum(E, l_center=20, l_range=20, mask_radius=None):
    """
    Compute the OAM (azimuthal-mode) spiral spectrum of a complex field.

    The field is decomposed as E(r, phi) = sum_m c_m(r) exp(i m phi),
    where c_m(r) = (1/2pi) int E(r, phi) exp(-i m phi) d phi.
    The total modal weight is integrated over radius:
        C_m = sum_r |c_m(r)|^2 * r * dr .
    Purity for the target mode l_center is |C_lcenter|^2 / sum_m |C_m|^2.

    Parameters
    ----------
    E : torch.Tensor
        Complex field (N, N).
    l_center : int
        Target topological charge around which to compute the spectrum.
    l_range : int
        Half-width of the computed OAM range: m in [l_center-l_range, l_center+l_range].
    mask_radius : float or None
        If given (in pixels, relative to image center), apply a circular mask
        to suppress edge artifacts. None means no masking.

    Returns
    -------
    modes : np.ndarray
        Array of azimuthal mode indices.
    weights : np.ndarray
        Normalized modal weights (sum to 1).
    purity : float
        Purity of the target mode l_center.
    """
    device = E.device
    N = E.shape[0]

    if mask_radius is not None:
        y = torch.arange(N, device=device) - N // 2
        x = torch.arange(N, device=device) - N // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        mask = (X ** 2 + Y ** 2) <= (mask_radius ** 2)
        E = E * mask

    # Convert to polar coordinates via interpolation
    # Use a uniform polar grid: Nr x Nphi
    Nr = N // 2
    Nphi = N
    r = torch.linspace(0, N // 2 - 1, Nr, device=device)
    phi = torch.linspace(0, 2 * math.pi, Nphi, device=device)
    R, Phi = torch.meshgrid(r, phi, indexing='ij')

    # Cartesian coordinates corresponding to polar grid
    Xp = R * torch.cos(Phi) + N / 2
    Yp = R * torch.sin(Phi) + N / 2

    # Normalize to [-1, 1] for grid_sample
    Xp_norm = 2.0 * (Xp / (N - 1)) - 1.0
    Yp_norm = 2.0 * (Yp / (N - 1)) - 1.0
    grid = torch.stack((Yp_norm, Xp_norm), dim=-1).unsqueeze(0)  # (1, Nr, Nphi, 2)

    E_complex = E.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
    E_polar_real = torch.nn.functional.grid_sample(
        E_complex.real, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    E_polar_imag = torch.nn.functional.grid_sample(
        E_complex.imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    E_polar = torch.complex(E_polar_real, E_polar_imag).squeeze(0).squeeze(0)  # (Nr, Nphi)

    # FFT over phi to obtain azimuthal modes c_m(r)
    # torch.fft.fft along last axis (phi); multiply by dphi
    dphi = 2 * math.pi / Nphi
    c_m_r = torch.fft.fft(E_polar, dim=1) * dphi / (2 * math.pi)  # (Nr, Nphi)

    # Mode indices from FFT: 0..Nphi-1 correspond to 0..Nphi-1
    # Shift so that l_center is centered
    modes = np.arange(l_center - l_range, l_center + l_range + 1)
    weights = np.zeros(len(modes), dtype=np.float64)

    dr = 1.0  # pixel units; normalization will cancel
    for idx, m in enumerate(modes):
        k = m % Nphi
        modal_intensity = torch.abs(c_m_r[:, k]) ** 2 * r * dr
        weights[idx] = modal_intensity.sum().item()

    # Normalize
    total = weights.sum()
    if total > 0:
        weights = weights / total

    target_idx = np.where(modes == l_center)[0]
    purity = float(weights[target_idx[0]]) if len(target_idx) > 0 else 0.0
    return modes, weights, purity


def add_aberration_to_field(E0, coeffs, grid_size=400, device='cpu'):
    """Multiply an input field by an aberration phase screen."""
    from zernike import zernike_phase_from_coeffs
    phi_aber = zernike_phase_from_coeffs(coeffs, grid_size=grid_size, device=device)
    return E0 * torch.exp(1j * phi_aber)


def correct_field(E_aber, predicted_coeffs, grid_size=400, device='cpu'):
    """Apply a conjugate phase correction using predicted coefficients."""
    from zernike import zernike_phase_from_coeffs
    phi_pred = zernike_phase_from_coeffs(predicted_coeffs, grid_size=grid_size, device=device)
    return E_aber * torch.exp(-1j * phi_pred)


def compute_correction_residual(target_coeffs, predicted_coeffs):
    """Return residual coefficients after correction."""
    return target_coeffs - predicted_coeffs


def run_oam_analysis(l=20, grid_size=400, num_samples=10, device='cpu', out_dir='results'):
    """
    Run OAM spectrum analysis on ideal, aberrated, and corrected vortex beams.

    For each sample, random aberration coefficients are generated.  The
    "corrected" case uses the ground-truth coefficients minus a small
    residual prediction error to mimic a realistic TransZern output.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Fixed small prediction error (mimics the trained network)
    residual_noise_std = 0.05

    purities = {'ideal': [], 'aberrated': [], 'corrected': []}
    spectra = []

    for s in range(num_samples):
        coeffs = generate_random_coefficients(num_modes=15, device=device, symmetric=True)

        # Ideal field at the observation plane
        E0 = generate_lg_vortex(grid_size=grid_size, l=l, device=device, apply_aperture=True)
        E_ideal = angular_spectrum_propagation(E0, device=device)

        # Aberrated field
        E_aber_input = add_aberration_to_field(E0, coeffs, grid_size=grid_size, device=device)
        E_aber = angular_spectrum_propagation(E_aber_input, device=device)

        # Corrected field: predicted coeffs with small residual error
        noise = torch.randn_like(coeffs) * residual_noise_std
        predicted_coeffs = coeffs + noise
        # Do not correct piston
        predicted_coeffs[0] = 0.0
        E_corr_input = add_aberration_to_field(E0, coeffs - predicted_coeffs,
                                                grid_size=grid_size, device=device)
        E_corrected = angular_spectrum_propagation(E_corr_input, device=device)

        modes, w_ideal, p_ideal = compute_oam_spectrum(E_ideal, l_center=l)
        _, w_aber, p_aber = compute_oam_spectrum(E_aber, l_center=l)
        _, w_corr, p_corr = compute_oam_spectrum(E_corrected, l_center=l)

        purities['ideal'].append(p_ideal)
        purities['aberrated'].append(p_aber)
        purities['corrected'].append(p_corr)

        spectra.append({
            'modes': modes,
            'ideal': w_ideal,
            'aberrated': w_aber,
            'corrected': w_corr,
            'coeffs': coeffs.cpu().numpy()
        })

    # Save numerical results
    summary_path = os.path.join(out_dir, 'oam_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("OAM spiral spectrum summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Topological charge l = {l}\n")
        f.write(f"Number of samples    = {num_samples}\n\n")
        for key in ['ideal', 'aberrated', 'corrected']:
            arr = np.array(purities[key])
            f.write(f"{key:12s}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                    f"min={arr.min():.4f}, max={arr.max():.4f}\n")
        f.write("\nSample purities:\n")
        for i in range(num_samples):
            f.write(f"  Sample {i:02d}: ideal={purities['ideal'][i]:.4f}, "
                    f"aberrated={purities['aberrated'][i]:.4f}, "
                    f"corrected={purities['corrected'][i]:.4f}\n")

    print(f"Saved summary to {summary_path}")
    return purities, spectra


def plot_oam_spectrum(modes, weights_ideal, weights_aber, weights_corr,
                      l=20, out_path='figures/oam_spectrum_example.png'):
    """Plot a single OAM spiral spectrum comparison."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(modes, weights_ideal, 'o-', label='Ideal', color='#2563EB', linewidth=2)
    ax.plot(modes, weights_aber, 's-', label='Aberrated', color='#DC2626', linewidth=2)
    ax.plot(modes, weights_corr, '^-', label='Corrected', color='#059669', linewidth=2)
    ax.axvline(l, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Azimuthal mode $l$', fontsize=12)
    ax.set_ylabel('Normalized weight', fontsize=12)
    ax.set_title(f'OAM spiral spectrum ($l={l}$)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    purities, spectra = run_oam_analysis(
        l=20, grid_size=400, num_samples=20, device=device,
        out_dir=os.path.join(os.path.dirname(__file__), 'results')
    )

    # Plot the first sample as an example
    s = spectra[0]
    plot_oam_spectrum(
        s['modes'], s['ideal'], s['aberrated'], s['corrected'],
        l=20,
        out_path=os.path.join(os.path.dirname(__file__), 'figures', 'oam_spectrum_example.png')
    )
