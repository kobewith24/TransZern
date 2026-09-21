import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 自动切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ZernikeNet
from dataset import PreGeneratedDataset
from lg_vortex import generate_vortex_beam
from propagation import angular_spectrum_propagation
from zernike import zernike_phase_from_coeffs


class PhysicalInformedLoss(nn.Module):
    """物理信息损失: 系数MSE + 强度一致性."""
    def __init__(self, coeff_weight=1.0, intensity_weight=0.3, epsilon=1e-8):
        super().__init__()
        self.coeff_weight = coeff_weight
        self.intensity_weight = intensity_weight
        self.epsilon = epsilon
        self.mse = nn.MSELoss()
        
        # Z1-Z14 系数范围 (根据数据生成逻辑)
        self.register_buffer('ranges', torch.tensor([
            15.0, 15.0, 12.0, 12.0, 12.0,
            9.0, 9.0, 9.0, 9.0,
            6.0, 6.0, 6.0, 6.0, 6.0
        ]))
    
    def forward(self, coeffs_pred, coeffs_true, I_distorted, device):
        # 系数归一化到 [0, 1] 再计算 MSE
        ranges = self.ranges.to(coeffs_pred.device)
        pred_norm = coeffs_pred[:, 1:15] / ranges
        true_norm = coeffs_true[:, 1:15] / ranges
        coeff_loss = self.mse(pred_norm, true_norm)
        intensity_loss = self._compute_intensity_loss(coeffs_pred, coeffs_true, I_distorted, device)
        return self.coeff_weight * coeff_loss + self.intensity_weight * intensity_loss
    
    def _compute_intensity_loss(self, coeffs_pred, coeffs_true, I_distorted, device):
        batch_size = coeffs_pred.size(0)
        losses = []
        
        grid_size, l, L, w = 400, 20, 6e-3, 2e-3
        wavelength, z = 532e-9, 300e-3
        
        for i in range(batch_size):
            E0 = generate_vortex_beam(grid_size, l, L, w, device=device)
            phi_true = zernike_phase_from_coeffs(coeffs_true[i], grid_size=grid_size, device=device)
            E_aberr = E0 * torch.exp(1j * phi_true)
            
            phi_pred = zernike_phase_from_coeffs(coeffs_pred[i], grid_size=grid_size, device=device)
            E_corrected = E_aberr * torch.exp(-1j * phi_pred)
            
            U_pred = angular_spectrum_propagation(E_corrected, wavelength, z, L, device=device)
            I_pred = torch.abs(U_pred) ** 2
            
            U_ideal = angular_spectrum_propagation(E0, wavelength, z, L, device=device)
            I_ideal = torch.abs(U_ideal) ** 2
            
            I_pred_n = (I_pred - I_pred.min()) / (I_pred.max() - I_pred.min() + self.epsilon)
            I_ideal_n = (I_ideal - I_ideal.min()) / (I_ideal.max() - I_ideal.min() + self.epsilon)
            
            losses.append(self.mse(I_pred_n, I_ideal_n))
        
        return torch.mean(torch.stack(losses))


def train_epoch(model, loader, optimizer, loss_fn, device, is_train=True):
    """训练/验证一个 epoch."""
    model.train() if is_train else model.eval()
    losses, grad_norms = [], []
    
    with torch.set_grad_enabled(is_train):
        for I_tensor, coeffs_tensor in tqdm(loader, desc='Train' if is_train else 'Val'):
            I_tensor = I_tensor.to(device)  # 单通道，无需 repeat
            coeffs_tensor = coeffs_tensor.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            coeffs_pred = model(I_tensor)
            total_loss = loss_fn(coeffs_pred, coeffs_tensor, I_tensor, device)
            
            if is_train:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norms.append(grad_norm.item())
                optimizer.step()
            
            losses.append(total_loss.item())
    
    return np.mean(losses), np.mean(grad_norms) if grad_norms else 0


def train_model(num_epochs=100, batch_size=16, num_modes=15, save_path=None, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ZernikeNet(num_modes=num_modes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    loss_fn = PhysicalInformedLoss(coeff_weight=1.0, intensity_weight=0.3)
    best_loss = float('inf')
    best_state = None
    start_epoch = 0
    
    # 断点续训
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        best_state = checkpoint['best_model_state']
        print(f"Resumed from checkpoint: {resume_path}, epoch {start_epoch}")
    
    writer = SummaryWriter(log_dir="ZernikeNet/logs_z", purge_step=start_epoch)
    
    full_dataset = PreGeneratedDataset(dataset_dir="ZernikeNet/datasets", device=device)
    train_ds = Subset(full_dataset, range(7000))
    val_ds = Subset(full_dataset, range(7000, 9000))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, grad_norm = train_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss, _ = train_epoch(model, val_loader, None, loss_fn, device, False)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.2e}, Grad={grad_norm:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', lr, epoch)
        writer.add_scalar('Gradient/norm', grad_norm, epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            if save_path:
                torch.save(best_state, save_path)
        
        if resume_path:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_model_state': best_state,
            }, resume_path)
        
        scheduler.step()
    
    if save_path and best_state:
        torch.save(best_state, save_path)
    writer.close()
    return model


if __name__ == "__main__":
    os.makedirs("ZernikeNet/models", exist_ok=True)
    save_path = "ZernikeNet/models/best_z.pth"
    resume_path = "ZernikeNet/models/checkpoint_z.pth"
    train_model(num_epochs=140, batch_size=16, num_modes=15, 
                save_path=save_path, resume_path=resume_path)
