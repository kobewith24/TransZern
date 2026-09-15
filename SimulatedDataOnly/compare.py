import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 先导入 SimulatedDataOnly 的 DirectDataset
sys.path.insert(0, 'SimulatedDataOnly')
from dataset import DirectDataset

# 再导入 Joint 的模块
sys.path.insert(0, 'Joint')
from model import JointAberrationCorrection, ZernikeNet
from utils import zernike_regularizer

# 临时导入 JointDataset，避免路径冲突
import importlib.util
spec = importlib.util.spec_from_file_location("joint_dataset", "Joint/dataset.py")
joint_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint_dataset_module)
JointDataset = joint_dataset_module.JointDataset


def load_joint_model(model_path, device='cuda'):
    """加载联合模型 (ConvertNet + ZernikeNet)."""
    model = JointAberrationCorrection().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_sim_only_model(model_path, device='cuda'):
    """加载仅用仿真数据训练的 ZernikeNet."""
    model = ZernikeNet(num_modes=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def compute_rmse(coeff_pred, coeff_true):
    """计算归一化 RMSE (Z1-Z14), per-sample."""
    ranges = torch.tensor([
        15.0, 15.0, 12.0, 12.0, 12.0,
        9.0, 9.0, 9.0, 9.0,
        6.0, 6.0, 6.0, 6.0, 6.0
    ]).to(coeff_pred.device)
    
    pred_norm = coeff_pred[:, 1:15] / ranges
    true_norm = coeff_true[:, 1:15] / ranges
    rmse = torch.sqrt(torch.mean((pred_norm - true_norm) ** 2, dim=1))
    return rmse


def compute_rmse_per_mode(coeff_pred, coeff_true):
    """计算每个 mode 的归一化 RMSE (Z1-Z14)."""
    ranges = torch.tensor([
        15.0, 15.0, 12.0, 12.0, 12.0,
        9.0, 9.0, 9.0, 9.0,
        6.0, 6.0, 6.0, 6.0, 6.0
    ]).to(coeff_pred.device)
    
    pred_norm = coeff_pred[:, 1:15] / ranges
    true_norm = coeff_true[:, 1:15] / ranges
    # 每个 mode 的 RMSE
    mse_per_mode = torch.mean((pred_norm - true_norm) ** 2, dim=0)
    rmse_per_mode = torch.sqrt(mse_per_mode)
    return rmse_per_mode


@torch.no_grad()
def evaluate_models(joint_model_path, sim_only_model_path, data_dir="SimulatedDataOnly/datasets"):
    """对比评估两个模型.
    
    With ConvertNet: 实采数据 -> ConvertNet -> ZernikeNet -> 系数
    Without ConvertNet: 实采数据 -> ZernikeNet(仅用仿真训练) -> 系数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 加载模型
    print("Loading models...")
    joint_model = load_joint_model(joint_model_path, device)
    sim_only_model = load_sim_only_model(sim_only_model_path, device)
    
    # 准备数据（使用 JointDataset 包含亮度归一化，与 test.py 一致）
    dataset = JointDataset(data_dir="Joint/datasets", device=device)
    n = len(dataset)
    # 使用与 test.py 相同的测试集划分: range(1400, 1700)
    test_ds = Subset(dataset, range(1700, 2000))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Test set size: {len(test_ds)} (real experimental data, indices 1400-1699)\n")
    
    # 评估
    joint_rmses_per_sample = []
    sim_only_rmses_per_sample = []
    joint_rmses_per_mode = []
    sim_only_rmses_per_mode = []
    joint_zernike_losses = []
    sim_only_zernike_losses = []
    
    ranges = torch.tensor([
        15.0, 15.0, 12.0, 12.0, 12.0,
        9.0, 9.0, 9.0, 9.0,
        6.0, 6.0, 6.0, 6.0, 6.0
    ]).to(device)
    
    for real_img, sim_img, coeffs in tqdm(test_loader, desc="Evaluating"):
        # With ConvertNet: 实采 -> ConvertNet -> ZernikeNet
        _, joint_pred = joint_model(real_img)
        joint_rmse_sample = compute_rmse(joint_pred, coeffs)
        joint_rmses_per_sample.extend(joint_rmse_sample.cpu().numpy())
        
        joint_rmse_mode = compute_rmse_per_mode(joint_pred, coeffs)
        joint_rmses_per_mode.append(joint_rmse_mode.cpu().numpy())
        
        joint_pred_norm = joint_pred[:, 1:15] / ranges
        true_norm = coeffs[:, 1:15] / ranges
        joint_zernike_loss = torch.mean((joint_pred_norm - true_norm) ** 2)
        joint_zernike_losses.append(joint_zernike_loss.item())
        
        # Without ConvertNet: 实采 -> ZernikeNet(仅用仿真训练)
        sim_only_pred = sim_only_model(real_img)
        sim_only_rmse_sample = compute_rmse(sim_only_pred, coeffs)
        sim_only_rmses_per_sample.extend(sim_only_rmse_sample.cpu().numpy())
        
        sim_only_rmse_mode = compute_rmse_per_mode(sim_only_pred, coeffs)
        sim_only_rmses_per_mode.append(sim_only_rmse_mode.cpu().numpy())
        
        sim_only_pred_norm = sim_only_pred[:, 1:15] / ranges
        sim_only_zernike_loss = torch.mean((sim_only_pred_norm - true_norm) ** 2)
        sim_only_zernike_losses.append(sim_only_zernike_loss.item())
    
    # 统计结果 (per-sample RMSE)
    joint_rmse_mean_sample = np.mean(joint_rmses_per_sample)
    joint_rmse_std_sample = np.std(joint_rmses_per_sample)
    sim_only_rmse_mean_sample = np.mean(sim_only_rmses_per_sample)
    sim_only_rmse_std_sample = np.std(sim_only_rmses_per_sample)
    
    # 统计结果 (per-mode RMSE average) - 与正文3.2节一致
    joint_rmses_per_mode = np.array(joint_rmses_per_mode)
    sim_only_rmses_per_mode = np.array(sim_only_rmses_per_mode)
    joint_rmse_mean_mode = np.mean(joint_rmses_per_mode)
    sim_only_rmse_mean_mode = np.mean(sim_only_rmses_per_mode)
    
    joint_zernike_mean = np.mean(joint_zernike_losses)
    sim_only_zernike_mean = np.mean(sim_only_zernike_losses)
    
    # 计算改进百分比 (使用 per-mode average，与正文一致)
    improvement_mode = (sim_only_rmse_mean_mode - joint_rmse_mean_mode) / sim_only_rmse_mean_mode * 100
    
    print("\n" + "="*60)
    print("ABlation Study Results: With vs Without ConvertNet")
    print("="*60)
    print("\n[Per-sample RMSE average]")
    print(f"{'Metric':<30} {'With ConvertNet':<20} {'Without ConvertNet':<20} {'Improvement':<15}")
    print("-"*85)
    print(f"{'Normalized RMSE':<30} {joint_rmse_mean_sample:.4f} ± {joint_rmse_std_sample:.4f}     {sim_only_rmse_mean_sample:.4f} ± {sim_only_rmse_std_sample:.4f}     -")
    print("\n[Per-mode RMSE average] (consistent with Section 3.2)")
    print(f"{'Metric':<30} {'With ConvertNet':<20} {'Without ConvertNet':<20} {'Improvement':<15}")
    print("-"*85)
    print(f"{'Normalized RMSE':<30} {joint_rmse_mean_mode:.4f}                  {sim_only_rmse_mean_mode:.4f}                  {improvement_mode:+.1f}%")
    print(f"{'Zernike MSE':<30} {joint_zernike_mean:.4f}                 {sim_only_zernike_mean:.4f}                 {((sim_only_zernike_mean-joint_zernike_mean)/sim_only_zernike_mean*100):+.1f}%")
    print(f"{'Test Samples':<30} {len(test_ds):<20} {len(test_ds):<20}")
    print("="*60)
    
    # 保存结果到 CSV
    results = {
        'Metric': ['Per-sample RMSE (mean)', 'Per-sample RMSE (std)', 'Per-mode RMSE (mean)', 'Zernike MSE', 'Test Samples'],
        'With ConvertNet': [f"{joint_rmse_mean_sample:.4f}", f"{joint_rmse_std_sample:.4f}", f"{joint_rmse_mean_mode:.4f}", f"{joint_zernike_mean:.4f}", str(len(test_ds))],
        'Without ConvertNet': [f"{sim_only_rmse_mean_sample:.4f}", f"{sim_only_rmse_std_sample:.4f}", f"{sim_only_rmse_mean_mode:.4f}", f"{sim_only_zernike_mean:.4f}", str(len(test_ds))],
        'Improvement': ["-", "-", f"{improvement_mode:+.1f}%", f"{((sim_only_zernike_mean-joint_zernike_mean)/sim_only_zernike_mean*100):+.1f}%", "-"]
    }
    df = pd.DataFrame(results)
    df.to_csv("SimulatedDataOnly/results/ablation_comparison.csv", index=False)
    print("\nResults saved to: SimulatedDataOnly/results/ablation_comparison.csv")
    
    # 保存详细对比 (per-mode)
    modes = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14']
    detailed_mode = pd.DataFrame({
        'Mode': modes,
        'Joint_RMSE': np.mean(joint_rmses_per_mode, axis=0),
        'SimOnly_RMSE': np.mean(sim_only_rmses_per_mode, axis=0),
    })
    detailed_mode.to_csv("SimulatedDataOnly/results/detailed_per_mode.csv", index=False)
    print("Per-mode comparison saved to: SimulatedDataOnly/results/detailed_per_mode.csv")
    
    return joint_rmse_mean_mode, sim_only_rmse_mean_mode


if __name__ == "__main__":
    # 默认路径
    joint_model = "Joint/models/final_joint.pth"
    sim_only_model = "SimulatedDataOnly/models/final_sim_only.pth"
    
    # 检查模型是否存在
    if not os.path.exists(joint_model):
        print(f"Joint model not found: {joint_model}")
        print("Please train the joint model first.")
        sys.exit(1)
    
    if not os.path.exists(sim_only_model):
        print(f"Sim-only model not found: {sim_only_model}")
        print("Please train the sim-only model first using: python SimulatedDataOnly/train_sim_only.py")
        sys.exit(1)
    
    evaluate_models(joint_model, sim_only_model)
