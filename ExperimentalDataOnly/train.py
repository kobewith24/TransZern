import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

# 自动切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ZernikeNet
from dataset import JointDataset


class DirectLoss(nn.Module):
    """直接回归损失: 只对Zernike系数做MSE."""
    def __init__(self, zernike_w=1.0):
        super().__init__()
        self.zernike_w = zernike_w
        self.mse = nn.MSELoss()
        
        # Z1-Z14 系数范围
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
        return zernike_loss


def train_epoch(model, loader, optimizer, loss_fn, device, is_train=True):
    """训练/验证一个epoch."""
    model.train() if is_train else model.eval()
    losses = []
    
    with torch.set_grad_enabled(is_train):
        for real_img, sim_img, coeffs in tqdm(loader, desc='Train' if is_train else 'Val'):
            real_img = real_img.to(device)
            coeffs = coeffs.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            # 直接用实验光强图回归系数
            coeff_pred = model(real_img)
            loss = loss_fn(coeff_pred, coeffs)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            losses.append(loss.item())
    
    return np.mean(losses)


def train_direct(num_epochs=140, batch_size=4, save_dir="ExperimentalDataOnly/models"):
    """直接训练ZernikeNet: 实验光强 -> 系数 (无ConvertNet)."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Training ZernikeNet directly on experimental intensity (no ConvertNet)...")
    
    model = ZernikeNet(num_modes=15).to(device)
    loss_fn = DirectLoss()
    
    # 使用Joint数据集，但只用实验光强图
    full_dataset = JointDataset(data_dir="Joint/datasets", device=device)
    n = len(full_dataset)
    train_ds = Subset(full_dataset, range(int(0.7 * n)))
    val_ds = Subset(full_dataset, range(int(0.7 * n), int(0.85 * n)))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    writer = SummaryWriter(log_dir="ExperimentalDataOnly/logs")
    best_val = float('inf')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss = train_epoch(model, val_loader, None, loss_fn, device, False)
        
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_direct.pth"))
        
        # 保存checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val': best_val,
        }, os.path.join(save_dir, "checkpoint.pth"))
        
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(save_dir, "final_direct.pth"))
    writer.close()
    print(f"\nTraining completed! Best val loss: {best_val:.6f}")
    return model


if __name__ == "__main__":
    os.makedirs("ExperimentalDataOnly/models", exist_ok=True)
    os.makedirs("ExperimentalDataOnly/logs", exist_ok=True)
    os.makedirs("ExperimentalDataOnly/results", exist_ok=True)
    
    checkpoint_path = "ExperimentalDataOnly/models/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        ans = input("Checkpoint found. Resume training? (y/n): ")
        if ans.lower() == 'y':
            # 简化版续训逻辑
            print("Resuming not fully implemented, starting from scratch...")
    
    train_direct()
