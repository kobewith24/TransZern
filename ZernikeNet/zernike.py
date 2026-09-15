import torch
import math

def zernike_radial(n, m, rho):
    R = torch.zeros_like(rho)
    if (n - m) % 2 == 0:
        k_max = (n - m) // 2
        for k in range(k_max + 1):
            term1 = (n + m) // 2 - k
            term2 = (n - m) // 2 - k
            if term1 >= 0 and term2 >= 0:
                numerator = (-1)**k * math.factorial(n - k)
                denominator = (
                    math.factorial(k) *
                    math.factorial(term1) *
                    math.factorial(term2)
                )
                R = R + numerator / denominator * rho**(n - 2*k)
    return R

def zernike_polynomial(n, m, rho, theta):
    if m >= 0:
        Z = zernike_radial(n, m, rho) * torch.cos(m * theta)
    else:
        Z = zernike_radial(n, -m, rho) * torch.sin(-m * theta)
    return Z

def generate_phase_map(coefficients, grid_size=2000, device='cpu'):
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2)
    Theta = torch.atan2(Y, X)
    phase_map = torch.zeros_like(R)
    index = 0
    n = 0
    while index < len(coefficients):
        for m in range(-n, n+1, 2):
            if index < len(coefficients):
                phase_map = phase_map + coefficients[index] * zernike_polynomial(n, m, R, Theta)
                index += 1
        n += 1
    phase_map = (phase_map + math.pi) % (2*math.pi)
    return phase_map

def zernike_phase_from_coeffs(coeffs, grid_size=400, device='cpu'):
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.to(device).float()
    else:
        coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    phase = generate_phase_map(coeffs, grid_size=grid_size, device=device)
    return phase

def generate_aberration_phase(num_modes=15, grid_size=400, device='cpu'):
    coeffs = torch.zeros(num_modes, device=device)
    coeffs[0] = 0
    
    # 1阶倾斜 (Z1-Z2): (0,15)
    coeffs[1:3] = torch.empty(2, device=device).uniform_(0, 15)
    
    # 2阶离焦、像散 (Z3-Z5): (0,12)
    coeffs[3:6] = torch.empty(3, device=device).uniform_(0, 12)
    
    # 3阶彗差 (Z6-Z9): (0,9)
    coeffs[6:10] = torch.empty(4, device=device).uniform_(0, 9)
    
    # 4阶高阶 (Z10-Z14): (0,6)
    coeffs[10:15] = torch.empty(5, device=device).uniform_(0, 6)
    phase = zernike_phase_from_coeffs(coeffs, grid_size=grid_size, device=device)
    return phase, coeffs