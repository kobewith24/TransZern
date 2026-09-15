import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, Dataset

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, 'Joint')
from model import ZernikeNet
from utils import zernike_regularizer

sys.path.insert(0, 'SimulatedDataOnly')
from dataset import DirectDataset


class SimOnlyLoss(nn.Module):
    """仅用仿真数据训练 ZernikeNet 的损失."""
    def __init__(self, zernike_w=1.0, reg_w=1e-3):
        super().__init__()
        self.zernike_w = zernike_w
        self.reg_w = reg_w
        self.mse = nn.MSELoss()
        
        self.register_buffer('ranges', torch.tensor([
            15.0, 15.0, 12.0, 12.0, 12.0,
            9.0, 9.0, 9.0, 9.0,
            6.0, 6.0, 6.0, 6.0, 6.0
        ]))
    
    def forward(self, coeff_pred, coeff_true):
        ranges = self.ranges.to(coeff_pred.device)
        pred_norm = coeff_pred[:, 1:15] / ranges
        true_norm = coeff_true[:, 1:15] / ranges
        zernike_loss = self.mse(pred_norm, true_norm)
        reg_loss = zernike_regularizer(coeff_pred, coeff_pred.device)
        total = self.zernike_w * zernike_loss + self.reg_w * reg_loss
        losses = {
            'total': total.item(),
            'zernike': zernike_loss.item(),
            'reg': reg_loss.item()
        }
        return total, losses


class SimOnlyDataset(Dataset):
    """仅用仿真数据训练的数据集."""
    def __init__(self, data_dir="Joint/datasets", device='cpu'):
        self.data_dir = data_dir
        self.device = device

        self.sim_dir = os.path.join(data_dir, "sim_intensity")
        self.csv_dir = os.path.join(data_dir, "csv")

        self.sim_files = sorted([f for f in os.listdir(self.sim_dir) if f.endswith('.png')])
        self.csv_files = sorted([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')])

        min_len = min(len(self.sim_files), len(self.csv_files))
        self.sim_files = self.sim_files[:min_len]
        self.csv_files = self.csv_files[:min_len]
    
    def _load_image(self, path):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img.astype('float32') / 255.0).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def _load_coefficients(self, path):
        import pandas as pd
        df = pd.read_csv(path, header=None)
        coeffs = torch.from_numpy(df.values.flatten().astype('float32'))
        if len(coeffs) < 15:
            padded_coeffs = torch.zeros(15, device=self.device)
            padded_coeffs[:len(coeffs)] = coeffs
            coeffs = padded_coeffs
        elif len(coeffs) > 15:
            coeffs = coeffs[:15]
        return coeffs.to(self.device)
    
    def __len__(self):
        return len(self.sim_files)
    
    def __getitem__(self, idx):
        sim_path = os.path.join(self.sim_dir, self.sim_files[idx])
        sim_img = self._load_image(sim_path)
        
        csv_path = os.path.join(self.csv_dir, self.csv_files[idx])
        coeffs = self._load_coefficients(csv_path)
        
        return sim_img, coeffs


def train_epoch(model, loader, optimizer, loss_fn, clip_norm=1.0):
    model.train()
    losses_list = []
    
    for sim_img, coeffs in tqdm(loader, desc='Train'):
        optimizer.zero_grad()
        coeff_pred = model(sim_img)
        total_loss, losses = loss_fn(coeff_pred, coeffs)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        losses_list.append(losses)
    
    return {k: np.mean([l[k] for l in losses_list]) for k in losses_list[0]}


@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    losses_list = []
    
    for sim_img, coeffs in loader:
        coeff_pred = model(sim_img)
        _, losses = loss_fn(coeff_pred, coeffs)
        losses_list.append(losses)
    
    return {k: np.mean([l[k] for l in losses_list]) for k in losses_list[0]}


def train_sim_only(num_epochs=140, batch_size=4, save_dir="SimulatedDataOnly/models"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ZernikeNet(num_modes=15).to(device)
    loss_fn = SimOnlyLoss()
    
    # 仅用仿真数据训练
    full_dataset = SimOnlyDataset(data_dir="Joint/datasets", device=device)
    n = len(full_dataset)
    train_ds = Subset(full_dataset, range(int(0.7 * n)))
    val_ds = Subset(full_dataset, range(int(0.7 * n), int(0.85 * n)))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    writer = SummaryWriter(log_dir="SimulatedDataOnly/logs_sim")
    best_val = float('inf')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    print("\n========== Training ZernikeNet on Simulation Data Only ==========")
    
    for epoch in range(num_epochs):
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn)
        val_losses = validate(model, val_loader, loss_fn)
        
        writer.add_scalars('Loss', {
            'train': train_losses['total'],
            'val': val_losses['total']
        }, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train={train_losses['total']:.6f}(z={train_losses['zernike']:.4f}), "
              f"Val={val_losses['total']:.6f}(z={val_losses['zernike']:.4f})")
        
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_sim_only.pth"))
            print(f"  *** Best model saved (val={best_val:.6f})")
        
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val': best_val,
        }, os.path.join(save_dir, "checkpoint_sim.pth"))
        
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(save_dir, "final_sim_only.pth"))
    writer.close()
    print(f"\nTraining completed! Best val loss: {best_val:.6f}")
    return model


if __name__ == "__main__":
    os.makedirs("SimulatedDataOnly/models", exist_ok=True)
    os.makedirs("SimulatedDataOnly/logs_sim", exist_ok=True)
    
    checkpoint_path = "SimulatedDataOnly/models/checkpoint_sim.pth"
    if os.path.exists(checkpoint_path):
        ans = input("\nCheckpoint found. Resume training? (y/n): ")
        if ans.lower() == 'y':
            # 简化版恢复训练
            print("Resuming not implemented, starting fresh...")
            train_sim_only()
        else:
            train_sim_only()
    else:
        train_sim_only()
