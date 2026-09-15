import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append('.')
from model import ZernikeNet
from dataset import JointDataset


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join("ExperimentalDataOnly/models", "final_direct.pth")
    model = ZernikeNet(num_modes=15).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    os.makedirs("ExperimentalDataOnly/results", exist_ok=True)

    full_dataset = JointDataset(data_dir="Joint/datasets", device=device)
    test_dataset = Subset(full_dataset, range(1700, 2000))    # 最后300个样本
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    coeffs_file_path = os.path.join('ExperimentalDataOnly/results', 'coeffs_direct.txt')
    with open(coeffs_file_path, 'w') as f:
        f.write("Sample\tType\tZ0\tZ1\tZ2\tZ3\tZ4\tZ5\tZ6\tZ7\tZ8\tZ9\tZ10\tZ11\tZ12\tZ13\tZ14\n")
    
    ranges = np.array([15.0, 15.0, 12.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    all_true_norm = []
    all_pred_norm = []

    for idx, (real_img, sim_img, true_coeffs) in enumerate(test_loader):
        with torch.no_grad():
            pred_coeffs = model(real_img)
            
            true_coeffs_np = true_coeffs.cpu().numpy()[0]
            pred_coeffs_np = pred_coeffs.cpu().numpy()[0]
            
            with open(coeffs_file_path, 'a') as f:
                true_values = [f"{x:.6f}" for x in true_coeffs_np]
                f.write(f"{idx}\tTrue\t" + "\t".join(true_values) + "\n")
                
                pred_values = [f"{x:.6f}" for x in pred_coeffs_np]
                f.write(f"{idx}\tPred\t" + "\t".join(pred_values) + "\n")
            
            true_norm = true_coeffs_np[1:15] / ranges
            pred_norm = pred_coeffs_np[1:15] / ranges
            
            all_true_norm.append(true_norm)
            all_pred_norm.append(pred_norm)
    
    all_true_norm = np.array(all_true_norm)
    all_pred_norm = np.array(all_pred_norm)
    
    mse_after_per_mode = np.mean((all_true_norm - all_pred_norm) ** 2, axis=0)
    rmse_after_per_mode = np.sqrt(mse_after_per_mode)
    
    mse_before_per_mode = np.mean(all_true_norm ** 2, axis=0)
    rmse_before_per_mode = np.sqrt(mse_before_per_mode)
    
    print("\n" + "="*70)
    print("RMSE per Zernike Mode (Normalized) - Direct Baseline")
    print("="*70)
    print(f"{'Mode':<10}{'Before Correction':<20}{'After Correction':<20}{'Improvement':<15}")
    print("-"*70)
    
    modes = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 
             'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14']
    
    for i, mode in enumerate(modes):
        improvement = (1 - rmse_after_per_mode[i] / rmse_before_per_mode[i]) * 100
        print(f"{mode:<10}{rmse_before_per_mode[i]:<20.6f}{rmse_after_per_mode[i]:<20.6f}{improvement:<15.2f}%")
    
    avg_rmse = np.mean(rmse_after_per_mode)
    print(f"\nAverage RMSE: {avg_rmse:.6f}")
    
    rmse_file_path = os.path.join('ExperimentalDataOnly/results', 'rmse_per_mode.txt')
    with open(rmse_file_path, 'w') as f:
        f.write("Mode\tBefore_Correction\tAfter_Correction\tImprovement_Percent\n")
        for i, mode in enumerate(modes):
            improvement = (1 - rmse_after_per_mode[i] / rmse_before_per_mode[i]) * 100
            f.write(f"{mode}\t{rmse_before_per_mode[i]:.6f}\t{rmse_after_per_mode[i]:.6f}\t{improvement:.2f}\n")
    print(f"\nPer-mode RMSE saved to: {rmse_file_path}")


if __name__ == "__main__":
    test()
