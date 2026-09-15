import torch


# 生成理想涡旋光束
def generate_vortex_beam(grid_size=400, l=20, L=6e-3, w=2e-3, device='cpu'):    # noqa: E741
    dx = L / grid_size
    dy = dx
    x = dx * (torch.arange(grid_size, device=device) - grid_size // 2)
    y = dy * (torch.arange(grid_size, device=device) - grid_size // 2)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    phase = l * torch.atan2(Y, X)
    
    # 生成高斯包络: 模拟涡旋光束强度
    gauss_envelope = torch.exp(-(X**2 + Y**2) / w**2)
    
    # 创建圆形孔径(软边)
    aperture_radius = L * 0.4
    aperture = torch.exp(-(X**2 + Y**2) / aperture_radius**2)
    
    E = gauss_envelope * aperture * torch.exp(1j * phase)

    return E



# def generate_vortex_beam(grid_size=400, l=20, L=6e-3, w=2e-3, device='cpu'):    # noqa: E741
#     dx = L / grid_size
#     dy = dx
#     x = dx * (torch.arange(grid_size, device=device) - grid_size // 2)
#     y = dy * (torch.arange(grid_size, device=device) - grid_size // 2)
#     X, Y = torch.meshgrid(x, y, indexing='ij')

#     phase = l * torch.atan2(Y, X)
#     E = torch.exp(-(X**2 + Y**2) / w**2) * torch.exp(1j * phase)

#     return E