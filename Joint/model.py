import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


class ConvBlock(nn.Module):
    """(Conv -> BN -> LeakyReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class ConvertNet(nn.Module):
    """U-Net: 实验光强 -> 仿真光强.
    
    输入: [B, 1, 400, 400]
    输出: [B, 1, 400, 400]
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        chs = [64, 128, 256, 512, 1024]
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.bottleneck = ConvBlock(chs[3], chs[4])
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.skip4 = nn.Conv2d(chs[3], chs[3], 1)
        self.dec4 = ConvBlock(chs[4], chs[3])
        
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.skip3 = nn.Conv2d(chs[2], chs[2], 1)
        self.dec3 = ConvBlock(chs[3], chs[2])
        
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.skip2 = nn.Conv2d(chs[1], chs[1], 1)
        self.dec2 = ConvBlock(chs[2], chs[1])
        
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.skip1 = nn.Conv2d(chs[0], chs[0], 1)
        self.dec1 = ConvBlock(chs[1], chs[0])
        
        # Output (双层)
        self.out = nn.Sequential(
            nn.Conv2d(chs[0], 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, self.skip4(e4)], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.skip3(e3)], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.skip2(e2)], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.skip1(e1)], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


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


class JointAberrationCorrection(nn.Module):
    """联合像差校正框架: ConvertNet + ZernikeNet."""
    def __init__(self, convert_net=None, zernike_net=None):
        super().__init__()
        self.convert_net = convert_net if convert_net is not None else ConvertNet()
        self.zernike_net = zernike_net if zernike_net is not None else ZernikeNet()
        
    def set_freeze(self, convert=False, zernike=False):
        """设置网络冻结状态."""
        for param in self.convert_net.parameters():
            param.requires_grad = not convert
        for param in self.zernike_net.parameters():
            param.requires_grad = not zernike
    
    def forward(self, x):
        """
        Args:
            x: 实验光强 [B, 1, H, W]
        Returns:
            sim_intensity: 转换后的仿真光强 [B, 1, H, W]
            coeffs: Zernike 系数 [B, 15]
        """
        sim_intensity = self.convert_net(x)
        coeffs = self.zernike_net(sim_intensity)
        return sim_intensity, coeffs
    
    def load_pretrained(self, convert_path, zernike_path, device='cpu'):
        """加载预训练权重."""
        print("\n=== Loading Pretrained Weights ===")
        
        for name, path, net in [('ConvertNet', convert_path, self.convert_net),
                                ('ZernikeNet', zernike_path, self.zernike_net)]:
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location=device, weights_only=True)
                    missing, unexpected = net.load_state_dict(state_dict, strict=False)
                    print(f"  Loaded {name}: {path}")
                    if missing:
                        print(f"    Missing keys: {missing}")
                    if unexpected:
                        print(f"    Unexpected keys: {unexpected}")
                except Exception as e:
                    print(f"  Failed to load {name}: {e}")
            else:
                print(f"  {name} weights not found: {path}")
