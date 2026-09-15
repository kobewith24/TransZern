import torch
import torch.nn as nn
import torchvision.models as models


class ZernikeNet(nn.Module):
    """基于 ResNet-18 的 Zernike 系数回归网络.
    
    输入: 单通道灰度强度图 [B, 1, H, W]
    输出: 15 维 Zernike 系数 [B, 15] (Z0 固定为 0)
    """
    def __init__(self, num_modes=15):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        
        # 修改第一层为单通道输入
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 回归头: 输出 14 维 (Z1-Z14)，Z0 固定为 0
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_modes - 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        coeffs_14 = self.backbone(x)  # [B, 14]
        batch_size = coeffs_14.size(0)
        z0 = torch.zeros(batch_size, 1, device=coeffs_14.device)
        coeffs = torch.cat([z0, coeffs_14], dim=1)  # [B, 15]
        return coeffs


def zernike_regularizer(coeffs, device='cpu'):
    """Zernike 系数物理正则化: 高阶系数权重衰减."""
    weights = torch.exp(-torch.arange(coeffs.size(1), device=device) / 6.0)
    return torch.mean((coeffs ** 2) * weights)
