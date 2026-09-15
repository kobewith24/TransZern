import torch
import torch.nn as nn
import torch.nn.functional as F


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
