import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from model import ConvertNet

# 自动切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# class NormalizedMSELoss(nn.Module):
#     def __init__(self, epsilon=1e-8):
#         super().__init__()
#         self.epsilon = epsilon
        
#     def forward(self, pred, target):
#         # 归一化到[0, 1]
#         pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + self.epsilon)
#         target_norm = (target - target.min()) / (target.max() - target.min() + self.epsilon)
#         return F.mse_loss(pred_norm, target_norm)
class NormalizedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8, brightness_weight=0.3):
        super().__init__()
        self.epsilon = epsilon
        self.brightness_weight = brightness_weight
        
    def forward(self, pred, target):
        # 归一化MSE损失
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + self.epsilon)
        target_norm = (target - target.min()) / (target.max() - target.min() + self.epsilon)
        normalized_mse = F.mse_loss(pred_norm, target_norm)
        
        # 亮度匹配损失
        pred_brightness = torch.mean(pred)
        target_brightness = torch.mean(target)
        brightness_loss = F.mse_loss(pred_brightness, target_brightness)

        total_loss = normalized_mse + self.brightness_weight * brightness_loss
        
        return total_loss


def train_model(num_epochs=100, batch_size=8, save_path=None, resume_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ConvertNet().to(device)
    
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # 余弦退火 + 热重启
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    loss_fn = NormalizedMSELoss()
    best_loss = float('inf')
    best_model_state = None

    start_epoch = 0
    # 断点续训: 加载checkpoint
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        best_model_state = checkpoint['best_model_state']
        print(f"Resumed from checkpoint: {resume_path}, epoch {start_epoch}")

    writer = SummaryWriter(log_dir="ConvertNet/logs_u", purge_step=start_epoch)
    
    # 700张作为训练集, 200张作为验证集, 100张作为测试集
    full_dataset = RealToSimDataset()
    train_dataset = Subset(full_dataset, range(700))    # 1600 200 200
    val_dataset = Subset(full_dataset, range(700, 900))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []
        grad_norms = []
        for real_tensor, sim_tensor in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            real_tensor = real_tensor.to(device)
            sim_tensor = sim_tensor.to(device)
            optimizer.zero_grad()
            sim_pred = model(real_tensor)
            loss = loss_fn(sim_pred, sim_tensor)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            grad_norms.append(grad_norm.item())
        mean_train = np.mean(train_losses)
        mean_grad_norm = np.mean(grad_norms)
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        val_losses = []
        with torch.no_grad():
            for real_tensor, sim_tensor in val_loader:
                real_tensor = real_tensor.to(device)
                sim_tensor = sim_tensor.to(device)
                sim_pred = model(real_tensor)
                loss = loss_fn(sim_pred, sim_tensor)
                val_losses.append(loss.item())
        
        mean_val = np.mean(val_losses)
        
        print(f"Epoch {epoch + 1}, Train loss={mean_train:.6f}, Val loss={mean_val:.6f}, Lr={current_lr:.2e}, Grad={mean_grad_norm:.4f}.")

        writer.add_scalar('Loss/train', mean_train, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Gradient/norm', mean_grad_norm, epoch)
        writer.add_scalar('Loss/val', mean_val, epoch)

        if mean_val < best_loss:
            best_loss = mean_val
            best_model_state = model.state_dict()
            if save_path is not None:
                torch.save(best_model_state, save_path)

        if resume_path is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_model_state': best_model_state,
            }
            torch.save(checkpoint, resume_path)
        
        scheduler.step()
        
    if save_path is not None and best_model_state is not None:
        torch.save(best_model_state, save_path)
    writer.close()
    return model


# dataset
class RealToSimDataset(torch.utils.data.Dataset):
    def __init__(self, real_int_dir="ConvertNet/datasets/real_intensity", sim_int_dir="ConvertNet/datasets/sim_intensity"):
        self.real_int_dir = real_int_dir
        self.sim_int_dir = sim_int_dir
        self.real_files = sorted([f for f in os.listdir(real_int_dir) if f.endswith('.png')])
        self.sim_files = sorted([f for f in os.listdir(sim_int_dir) if f.endswith('.png')])
        assert len(self.real_files) == len(self.sim_files), "Real and simulated image counts don't match"

        real_numbers = [f.split('_')[-1].split('.')[0] for f in self.real_files]
        sim_numbers = [f.split('_')[-1].split('.')[0] for f in self.sim_files]
        assert all(r == s for r, s in zip(real_numbers, sim_numbers)), "File numbers don't match"
    
    def __len__(self):
        return len(self.real_files)
    
    def __getitem__(self, idx):
        real_path = os.path.join(self.real_int_dir, self.real_files[idx])
        real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        real_tensor = torch.from_numpy(real_img.astype('float32') / 255.0).unsqueeze(0)
        sim_path = os.path.join(self.sim_int_dir, self.sim_files[idx])
        sim_img = cv2.imread(sim_path, cv2.IMREAD_GRAYSCALE)
        sim_tensor = torch.from_numpy(sim_img.astype('float32') / 255.0).unsqueeze(0)
        # 亮度归一化: 将实采图像调整到与仿真图像相同的亮度水平
        real_tensor = self.normalize_brightness(real_tensor, sim_tensor)
        
        return real_tensor, sim_tensor
    
    # 将实采畸变光场的亮度调整到与仿真畸变光场相同的水平
    def normalize_brightness(self, real_tensor, sim_tensor):
        # 计算亮度比率
        real_brightness = torch.mean(real_tensor)
        sim_brightness = torch.mean(sim_tensor)
        
        if real_brightness > 1e-6:    # 避免除零
            brightness_ratio = sim_brightness / real_brightness
            normalized_real = real_tensor * brightness_ratio
            normalized_real = torch.clamp(normalized_real, 0, 1)
            return normalized_real
        else:
            return real_tensor


if __name__ == "__main__":
    os.makedirs("ConvertNet/models", exist_ok=True)
    save_path = os.path.join("ConvertNet/models", "best_u.pth")
    resume_path = os.path.join("ConvertNet/models", "checkpoint_u.pth")
    model = train_model(num_epochs=140, batch_size=8, save_path=save_path, resume_path=resume_path)