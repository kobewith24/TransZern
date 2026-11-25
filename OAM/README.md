# OAM Spiral Spectrum Analysis

This folder contains a self-contained implementation for computing the **orbital angular momentum (OAM) spiral spectrum** of simulated ideal, aberrated, and corrected vortex fields. The input is a Gaussian beam with an imposed helical phase and a soft-aperture factor.

## Why OAM spectrum?

For vortex beams, the Strehl ratio is only an approximate quality metric because the annular intensity peak is not at a fixed position. The OAM spiral spectrum decomposes the beam into azimuthal modes and reports the **mode purity** of the target topological charge, which is a physically more appropriate figure of merit for OAM-based systems.

## Files

- `zernike.py` — Zernike polynomial generation with signed coefficient ranges.
- `optics.py` — Gaussian input beam with helical-phase modulation and angular-spectrum propagation.
- `oam_spectrum.py` — OAM spiral-spectrum computation, simulation, and plotting.
- `run_oam_analysis.py` — Command-line entry point.
- `results/` — Numerical outputs (purity statistics).
- `figures/` — Generated plots.

## Usage

```bash
cd OAM
python run_oam_analysis.py --l 20 --samples 20 --device cuda
```

Options:
- `--l`: topological charge (default 20)
- `--samples`: number of random aberration realizations (default 20)
- `--device`: `cpu` or `cuda` (default `cpu`)
- `--grid-size`: simulation grid size (default 400)

## Output

- `results/oam_summary.txt` — Mean/std/min/max purity for ideal, aberrated, and corrected beams.
- `figures/oam_spectrum_l20.png` — Example OAM spectrum plot.

## Method

The complex field $E(r,\phi)$ is interpolated onto a polar grid and decomposed azimuthally via FFT:

$$c_m(r) = \frac{1}{2\pi} \int_0^{2\pi} E(r,\phi) e^{-i m \phi} \, d\phi$$

The total weight of mode $m$ is integrated over radius:

$$C_m = \int |c_m(r)|^2 \, r \, dr$$

Mode purity for target charge $l$:

$$P_l = \frac{C_l}{\sum_m C_m}$$

This OAM simulation uses the soft-aperture radius parameter $r_a = 0.45L$ (the synthetic training model uses $0.4L$). At $l=20$, the normalization sum covers $m=0,\ldots,40$. The 400-point angular grid includes both $0$ and $2\pi$, so the repeated sample introduces leakage in the discrete spectrum. The reported ideal purity of approximately 0.989 is the numerical reference from this procedure.

## Note

The simulated corrected field uses the conjugate phase constructed from the injected coefficients plus independent zero-mean Gaussian errors with a standard deviation of 0.052 times each mode’s coefficient half-range. To compute a corrected field using actual model predictions, pass the predicted coefficients to `correct_field()` in `oam_spectrum.py`.
