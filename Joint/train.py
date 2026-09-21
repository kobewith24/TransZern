import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# 自动切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import JointAberrationCorrection, ConvertNet
from dataset import JointDataset
from utils import zernike_regularizer


class JointLoss(nn.Module):
    """联合损失函数: 转换损失 + Zernike 损失 + 正则化."""
    def __init__(self, convert_w=1.0, zernike_w=1.0, reg_w=1e-3):
        super().__init__()
        self.convert_w = convert_w
        self.zernike_w = zernike_w
        self.reg_w = reg_w
        self.mse = nn.MSELoss()
        
        # Z1-Z14 系数范围
        self.register_buffer('ranges', torch.tensor([
            15.0, 15.0, 12.0, 12.0, 12.0,
            9.0, 9.0, 9.0, 9.0,
            6.0, 6.0, 6.0, 6.0, 6.0
        ]))
    
    def forward(self, sim_pred, sim_true, coeff_pred, coeff_true, phase='phase1'):
        convert_loss = self.mse(sim_pred, sim_true)
        
        if phase == 'phase1':
            total = self.convert_w * convert_loss
            losses = {'total': total.item(), 'convert': convert_loss.item()}
        else:
            # 忽略 Z0，并对 Z1-Z14 归一化
            ranges = self.ranges.to(coeff_pred.device)
            pred_norm = coeff_pred[:, 1:15] / ranges
            true_norm = coeff_true[:, 1:15] / ranges
            zernike_loss = self.mse(pred_norm, true_norm)
            reg_loss = zernike_regularizer(coeff_pred, coeff_pred.device)
            total = (self.convert_w * convert_loss + 
                    self.zernike_w * zernike_loss + 
                    self.reg_w * reg_loss)
            losses = {
                'total': total.item(),
                'convert': convert_loss.item(),
                'zernike': zernike_loss.item(),
                'reg': reg_loss.item()
            }
        return total, losses


def train_phase(model, loader, optimizer, loss_fn, phase, clip_norm=1.0):
    """训练一个 epoch."""
    model.train()
    losses_list = []
    
    for real_img, sim_img, coeffs in tqdm(loader, desc=f'Train [{phase}]'):
        optimizer.zero_grad()
        
        sim_pred, coeff_pred = model(real_img)
        total_loss, losses = loss_fn(sim_pred, sim_img, coeff_pred, coeffs, phase=phase)
        
        total_loss.backward()
        if phase == 'phase1':
            torch.nn.utils.clip_grad_norm_(model.convert_net.parameters(), clip_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        losses_list.append(losses)
    
    return {k: np.mean([l[k] for l in losses_list]) for k in losses_list[0]}


@torch.no_grad()
def validate(model, loader, loss_fn, phase):
    """验证."""
    model.eval()
    losses_list = []
    
    for real_img, sim_img, coeffs in loader:
        sim_pred, coeff_pred = model(real_img)
        _, losses = loss_fn(sim_pred, sim_img, coeff_pred, coeffs, phase=phase)
        losses_list.append(losses)
    
    return {k: np.mean([l[k] for l in losses_list]) for k in losses_list[0]}


def train_joint(num_epochs=60, batch_size=4, save_dir="Joint/models"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = JointAberrationCorrection().to(device)
    
    # 加载预训练权重
    convert_path = os.path.join(save_dir, "best_u.pth")
    zernike_path = os.path.join(save_dir, "best_z.pth")
    model.load_pretrained(convert_path, zernike_path, device)
    
    loss_fn = JointLoss()
    
    # 数据集
    full_dataset = JointDataset(data_dir="Joint/datasets", device=device)
    n = len(full_dataset)
    train_ds = Subset(full_dataset, range(int(0.7 * n)))
    val_ds = Subset(full_dataset, range(int(0.7 * n), int(0.9 * n)))    # 0.7 0.9
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    
    best_val = float('inf')
    
    # ========== Phase 1: 微调 ConvertNet ==========
    print("\n========== Phase 1: Fine-tune ConvertNet ==========")
    model.set_freeze(convert=False, zernike=True)
    
    phase1_epochs = 20
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=1e-6)
    
    for epoch in range(phase1_epochs):
        train_losses = train_phase(model, train_loader, optimizer, loss_fn, 'phase1')
        val_losses = validate(model, val_loader, loss_fn, 'phase1')
        
        print(f"Epoch {epoch+1}/{phase1_epochs}: "
              f"Train={train_losses['total']:.6f}, Val={val_losses['total']:.6f}")
        
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_phase1.pth"))
        
        scheduler.step()
    
    # ========== Phase 2: 联合训练 ==========
    print("\n========== Phase 2: Joint Training ==========")
    model.set_freeze(convert=False, zernike=False)
    
    phase2_epochs = num_epochs - phase1_epochs
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)
    
    for epoch in range(phase2_epochs):
        train_losses = train_phase(model, train_loader, optimizer, loss_fn, 'phase2')
        val_losses = validate(model, val_loader, loss_fn, 'phase2')
        
        print(f"Epoch {epoch+1}/{phase2_epochs}: "
              f"Train={train_losses['total']:.6f}(c={train_losses['convert']:.4f},z={train_losses['zernike']:.4f}), "
              f"Val={val_losses['total']:.6f}")
        
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_joint.pth"))
        
        # 保存 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val': best_val,
        }, os.path.join(save_dir, "checkpoint.pth"))
        
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(save_dir, "final_joint.pth"))
    print(f"\nTraining completed! Best val loss: {best_val:.6f}")
    return model


def test_pretrained():
    """测试 ConvertNet (在 CPU 上运行以节省显存)."""
    device = 'cpu'  # 显存不足时用 CPU
    path = "Joint/models/best_u.pth"
    
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return
    
    model = ConvertNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
    model.eval()
    
    dataset = JointDataset(data_dir="Joint/datasets", device=device)
    test_ds = Subset(dataset, range(min(3, len(dataset))))
    
    losses = []
    for i in range(len(test_ds)):
        real_img, sim_img, _ = test_ds[i]
        with torch.no_grad():
            pred = model(real_img.unsqueeze(0))
            losses.append(nn.MSELoss()(pred, sim_img.unsqueeze(0)).item())
    
    print(f"Mean ConvertNet Loss: {np.mean(losses):.6f}")


def resume_training(checkpoint_path="Joint/models/checkpoint.pth", num_epochs=20):
    """从 checkpoint 恢复训练."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found, starting from scratch...")
        return train_joint()
    
    print(f"Resuming from: {checkpoint_path}")
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = JointAberrationCorrection().to(device)
    model.load_state_dict(checkpoint['model_state'])
    
    model.set_freeze(convert=False, zernike=False)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    best_val = checkpoint['best_val']
    start_epoch = checkpoint['epoch'] + 1
    
    loss_fn = JointLoss()
    
    full_dataset = JointDataset(data_dir="Joint/datasets", device=device)
    n = len(full_dataset)
    train_ds = Subset(full_dataset, range(int(0.7 * n)))
    val_ds = Subset(full_dataset, range(int(0.7 * n), int(1.0 * n)))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_losses = train_phase(model, train_loader, optimizer, loss_fn, 'phase2')
        val_losses = validate(model, val_loader, loss_fn, 'phase2')
        
        print(f"Epoch {epoch+1}: Train={train_losses['total']:.6f}, Val={val_losses['total']:.6f}")
        
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            torch.save(model.state_dict(), os.path.join("Joint/models", "best_joint.pth"))
        
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val': best_val,
        }, os.path.join("Joint/models", "checkpoint.pth"))
        
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join("Joint/models", "final_joint.pth"))
    return model


if __name__ == "__main__":
    os.makedirs("Joint/models", exist_ok=True)
    os.makedirs("Joint/results", exist_ok=True)
    
    test_pretrained()
    
    checkpoint_path = "Joint/models/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        ans = input("\nCheckpoint found. Resume training? (y/n): ")
        if ans.lower() == 'y':
            resume_training(checkpoint_path)
        else:
            train_joint()
    else:
        train_joint()
