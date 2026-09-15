# TransZern

Official code for "TransZern: A sensorless aberration correction method for vortex beams via domain translation" (Optics and Lasers in Engineering).

TransZern is a two-stage sensorless adaptive-optics framework for Laguerre-Gaussian (LG) vortex beams. A U-Net-based **ConvertNet** translates experimental intensity patterns into their simulated counterparts, and a ResNet-18-based **ZernikeNet** regresses 14 Zernike coefficients ($Z_2 \sim Z_{15}$, piston $Z_1$ fixed to zero). The two networks are first pretrained individually and then fine-tuned jointly.

## Repository structure

| Directory | Paper section | Description |
|---|---|---|
| `ConvertNet/` | Sec. 2.2 | U-Net for real-to-sim intensity translation; also contains the angular-spectrum simulator (`simulate.py`, `propagation.py`, `zernike.py`, `lg_vortex.py`) used to generate synthetic training data |
| `ZernikeNet/` | Sec. 2.2 | ResNet-18 regression network trained on 10,000 synthetic samples; `get_datasets.py` generates the synthetic dataset |
| `Joint/` | Secs. 2.2, 2.3 | Joint two-stage training (Phase 1: ConvertNet only; Phase 2: end-to-end) and end-to-end inference/test |
| `ExperimentalDataOnly/` | Sec. 3.4 | Ablation baseline: ZernikeNet trained directly on experimental data, without ConvertNet |
| `SimulatedDataOnly/` | Sec. 3.4 | Ablation baseline: ZernikeNet trained only on simulated data, evaluated on experimental data |
| `WithoutPretraining/` | Sec. 3.4 | Ablation baseline: joint model trained from scratch without individual pretraining |
| `OAM/` | Sec. 3.3 | OAM spiral-spectrum decomposition and purity computation for ideal, aberrated, and corrected fields |
| `TimeCost/` | Sec. 3.2 | End-to-end inference latency measurement (12.3 ms/sample on an NVIDIA RTX A6000) |
| `figures/` | Secs. 3.2, 3.5 | Scripts reproducing the per-mode RMSE bar chart and the ConvertNet data-efficiency curve |

## Data

The datasets and trained model checkpoints are **not** distributed with this repository:

- ConvertNet: 1,000 paired experimental/simulated distorted intensities (700/200/100 train/val/test)
- ZernikeNet: 10,000 simulated intensity–coefficient pairs (7,000/2,000/1,000 train/val/test)
- Independent experimental test set: 300 intensity distributions per topological charge ($l = 10, 20, 30$)

They are available from the corresponding authors upon reasonable request.

## Reproduction pipeline

1. **Generate synthetic data** with `ZernikeNet/get_datasets.py` (angular-spectrum propagation; parameters in Sec. 2.1 of the paper).
2. **Pretrain ConvertNet**: `python train.py` in `ConvertNet/` (L1 loss on paired data).
3. **Pretrain ZernikeNet**: `python train.py` in `ZernikeNet/` (MSE loss on synthetic data).
4. **Joint training**: `python train.py` in `Joint/` (composite objective, Phase 1 + Phase 2).
5. **Evaluate**: `python test.py` in `Joint/` for coefficient regression and image fidelity; `OAM/run_oam_analysis.py` for OAM spiral-spectrum purity; `TimeCost/measure_time.py` for inference latency.

Training configuration (batch size 4, AdamW, weight decay $10^{-5}$, learning rates $10^{-4}$/$5\times10^{-5}$, 20 + 40 epochs) is described in Sec. 2.3 of the paper.

## Environment

- Python 3.x, PyTorch with CUDA
- See `requirements.txt`

## Citation

If you use this code, please cite:

> Q. Wang, P. Dai, X. Cheng, Z. Gao, C. Zhang, "TransZern: A sensorless aberration correction method for vortex beams via domain translation," *Optics and Lasers in Engineering*, 2026. (to appear)
