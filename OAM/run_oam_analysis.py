#!/usr/bin/env python3
"""
Main entry point for OAM spiral-spectrum analysis.

Example:
    python run_oam_analysis.py --l 20 --samples 20 --device cuda
"""

import argparse
import os
import sys

# Ensure the OAM package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oam_spectrum import run_oam_analysis, plot_oam_spectrum


def main():
    parser = argparse.ArgumentParser(description='OAM spiral spectrum analysis for TransZern')
    parser.add_argument('--l', type=int, default=20, help='Topological charge')
    parser.add_argument('--grid-size', type=int, default=400, help='Simulation grid size')
    parser.add_argument('--samples', type=int, default=20, help='Number of random samples')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Computation device')
    parser.add_argument('--out-dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, args.out_dir)
    fig_dir = os.path.join(base_dir, 'figures')

    purities, spectra = run_oam_analysis(
        l=args.l,
        grid_size=args.grid_size,
        num_samples=args.samples,
        device=args.device,
        out_dir=out_dir
    )

    # Plot the first sample
    s = spectra[0]
    plot_oam_spectrum(
        s['modes'], s['ideal'], s['aberrated'], s['corrected'],
        l=args.l,
        out_path=os.path.join(fig_dir, f'oam_spectrum_l{args.l}.png')
    )

    print("\nOAM purity summary:")
    for key in ['ideal', 'aberrated', 'corrected']:
        arr = [purities[key][i] for i in range(len(purities[key]))]
        import numpy as np
        print(f"  {key:12s}: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}")


if __name__ == "__main__":
    main()
