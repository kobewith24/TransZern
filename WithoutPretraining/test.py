import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys

sys.path.append('.')
from model import JointAberrationCorrection
from dataset import JointDataset
from utils import (
    generate_vortex_beam, zernike_phase_from_coeffs
)


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join("WithoutPretraining/models", "final_joint.pth")
    model = JointAberrationCorrection().to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    os.makedirs("WithoutPretraining/results", exist_ok=True)
    os.makedirs("WithoutPretraining/results/phases", exist_ok=True)

    full_dataset = JointDataset(data_dir="Joint/datasets", device=device)
    test_dataset = Subset(full_dataset, range(1700, 2000))    # 最后300个样本
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    coeffs_file_path = os.path.join('WithoutPretraining/results', 'coeffs_joint.txt')
    with open(coeffs_file_path, 'w') as f:
        f.write("Sample\tType\tZ0\tZ1\tZ2\tZ3\tZ4\tZ5\tZ6\tZ7\tZ8\tZ9\tZ10\tZ11\tZ12\tZ13\tZ14\n")
    
    # 物理参数
    grid_size = 2000    # 相位
    l = 20
    L = 6e-3
    w = 2e-3

    total_mse = 0.0
    total_samples = 0
    
    # 用于存储每个mode的误差，以便计算per-mode RMSE
    ranges = np.array([15.0, 15.0, 12.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    all_true_norm = []      # 存储所有样本的真实归一化系数
    all_pred_norm = []      # 存储所有样本的预测归一化系数

    for idx, (real_img, sim_img, true_coeffs) in enumerate(test_loader):
        with torch.no_grad():
            # 前向传播
            sim_pred, pred_coeffs = model(real_img)
            
            # 获取系数
            true_coeffs_np = true_coeffs.cpu().numpy()[0]
            pred_coeffs_np = pred_coeffs.cpu().numpy()[0]
            
            # 保存系数
            with open(coeffs_file_path, 'a') as f:
                true_values = [f"{x:.6f}" for x in true_coeffs_np]
                f.write(f"{idx}\tTrue\t" + "\t".join(true_values) + "\n")
                
                pred_values = [f"{x:.6f}" for x in pred_coeffs_np]
                f.write(f"{idx}\tPred\t" + "\t".join(pred_values) + "\n")
            
            # 计算归一化系数
            true_norm = true_coeffs_np[1:15] / ranges
            pred_norm = pred_coeffs_np[1:15] / ranges
            
            all_true_norm.append(true_norm)
            all_pred_norm.append(pred_norm)
            
            # 计算当前样本的MSE（所有modes的平均）
            mse_per_sample = np.mean((true_norm - pred_norm) ** 2)
            total_mse += mse_per_sample
            total_samples += 1
        
        # 每60个样本输出一次结果, 共5次
        if idx % 60 == 0:
            print(f"\nSample {idx}: ")
            print(f"True Zernike Coefficients: {[f'{x:.3f}' for x in true_coeffs_np[1:15]]}")
            print(f"Predicted Zernike Coefficients: {[f'{x:.3f}' for x in pred_coeffs_np[1:15]]}")

            E0 = generate_vortex_beam(grid_size, l, L, w, device=device)

            true_aberration_phase = zernike_phase_from_coeffs(
                torch.from_numpy(true_coeffs_np).to(device), 
                grid_size=grid_size, 
                device=device
            )

            pred_aberration_phase = zernike_phase_from_coeffs(
                torch.from_numpy(pred_coeffs_np).to(device), 
                grid_size=grid_size, 
                device=device
            )

            E_distorted = E0 * torch.exp(1j * true_aberration_phase)
            distorted_phase = torch.angle(E_distorted)

            E_corrected = E_distorted * torch.exp(-1j * pred_aberration_phase)
            corrected_phase = torch.angle(E_corrected)
            
            # 转换为numpy
            distorted_phase_np = distorted_phase.cpu().numpy()
            corrected_phase_np = corrected_phase.cpu().numpy()
            true_aberration_phase_np = true_aberration_phase.cpu().numpy()
            pred_aberration_phase_np = pred_aberration_phase.cpu().numpy()

            sample_phase_dir = os.path.join('WithoutPretraining/results/phases', f'sample_{idx}')
            os.makedirs(sample_phase_dir, exist_ok=True)

            distorted_normalized = ((distorted_phase_np - distorted_phase_np.min()) / (distorted_phase_np.max() - distorted_phase_np.min()) * 255).astype(np.uint8)
            distorted_save_path = os.path.join(sample_phase_dir, 'distorted_phase.png')
            cv2.imwrite(distorted_save_path, distorted_normalized)
            
            corrected_normalized = ((corrected_phase_np - corrected_phase_np.min()) / (corrected_phase_np.max() - corrected_phase_np.min()) * 255).astype(np.uint8)
            corrected_save_path = os.path.join(sample_phase_dir, 'corrected_phase.png')
            cv2.imwrite(corrected_save_path, corrected_normalized)
            
            true_aberration_normalized = ((true_aberration_phase_np - true_aberration_phase_np.min()) / (true_aberration_phase_np.max() - true_aberration_phase_np.min()) * 255).astype(np.uint8)
            true_aberration_save_path = os.path.join(sample_phase_dir, 'true_aberration_phase.png')
            cv2.imwrite(true_aberration_save_path, true_aberration_normalized)
            
            pred_aberration_normalized = ((pred_aberration_phase_np - pred_aberration_phase_np.min()) / (pred_aberration_phase_np.max() - pred_aberration_phase_np.min()) * 255).astype(np.uint8)
            pred_aberration_save_path = os.path.join(sample_phase_dir, 'predicted_aberration_phase.png')
            cv2.imwrite(pred_aberration_save_path, pred_aberration_normalized)

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes[0, 0].imshow(distorted_phase_np, cmap='gray')
            axes[0, 0].set_title('Distorted Phase', fontsize=12)
            axes[0, 0].axis('off')
            
            # Corrected Phase
            axes[0, 1].imshow(corrected_phase_np, cmap='gray')
            axes[0, 1].set_title('Corrected Phase', fontsize=12)
            axes[0, 1].axis('off')
            
            # True Aberration Phase
            axes[1, 0].imshow(true_aberration_phase_np, cmap='gray')
            axes[1, 0].set_title('True Aberration Phase', fontsize=12)
            axes[1, 0].axis('off')
            
            # Predicted Aberration Phase
            axes[1, 1].imshow(pred_aberration_phase_np, cmap='gray')
            axes[1, 1].set_title('Predicted Aberration Phase', fontsize=12)
            axes[1, 1].axis('off')
            
            plt.tight_layout()

            comparison_save_path = os.path.join(sample_phase_dir, 'phase_comparison.png')
            plt.savefig(comparison_save_path, dpi=150, bbox_inches='tight')
            plt.close()

    if total_samples > 0:
        avg_mse = total_mse / total_samples
        print(f"\nMean MSE(Z1-Z14): {avg_mse:.6f}")
        
        # 计算每个mode的RMSE（校正后：真实值 vs 预测值）
        all_true_norm = np.array(all_true_norm)
        all_pred_norm = np.array(all_pred_norm)
        
        mse_after_per_mode = np.mean((all_true_norm - all_pred_norm) ** 2, axis=0)
        rmse_after_per_mode = np.sqrt(mse_after_per_mode)
        
        # 计算每个mode的RMSE（校正前：真实值 vs 0）
        mse_before_per_mode = np.mean(all_true_norm ** 2, axis=0)
        rmse_before_per_mode = np.sqrt(mse_before_per_mode)
        
        # 输出结果
        print("\n" + "="*70)
        print("RMSE per Zernike Mode (Normalized)")
        print("="*70)
        print(f"{'Mode':<10}{'Before Correction':<20}{'After Correction':<20}{'Improvement':<15}")
        print("-"*70)
        
        modes = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 
                 'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14']
        
        for i, mode in enumerate(modes):
            improvement = (1 - rmse_after_per_mode[i] / rmse_before_per_mode[i]) * 100
            print(f"{mode:<10}{rmse_before_per_mode[i]:<20.6f}{rmse_after_per_mode[i]:<20.6f}{improvement:<15.2f}%")
        
        # 保存到文件
        rmse_file_path = os.path.join('WithoutPretraining/results', 'rmse_per_mode.txt')
        with open(rmse_file_path, 'w') as f:
            f.write("Mode\tBefore_Correction\tAfter_Correction\tImprovement_Percent\n")
            for i, mode in enumerate(modes):
                improvement = (1 - rmse_after_per_mode[i] / rmse_before_per_mode[i]) * 100
                f.write(f"{mode}\t{rmse_before_per_mode[i]:.6f}\t{rmse_after_per_mode[i]:.6f}\t{improvement:.2f}\n")
        print(f"\nPer-mode RMSE saved to: {rmse_file_path}")
        
    else:
        print("Error: no test samples for calculating MSE!")


if __name__ == "__main__":
    test()