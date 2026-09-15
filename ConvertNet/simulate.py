import torch
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from lg_vortex import generate_vortex_beam
from propagation import angular_spectrum_propagation
from zernike import zernike_phase_from_coeffs


def simulate_intensity_from_zernike_coeffs():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs("ConvertNet/datasets/sim_intensity", exist_ok=True)
    coeff_dir = "ConvertNet/datasets/csv"
    coeff_files = sorted([f for f in os.listdir(coeff_dir) if f.startswith('zernike_coefs_') and f.endswith('.csv')])
    
    grid_size_high_res = 2000
    grid_size_low_res = 400
    wavelength = 532e-9
    z = 300e-3
    L = 6e-3
    w = 2e-3
    l = 20
    
    for coeff_file in tqdm(coeff_files, desc="Simulating intensity from Zernike coefficients"):
        file_num = coeff_file.split('_')[-1].split('.')[0]
        coeff_path = os.path.join(coeff_dir, coeff_file)
        df = pd.read_csv(coeff_path, header=None)
        coeffs = torch.from_numpy(df.values.flatten().astype('float32')).to(device)

        if len(coeffs) < 15:
            padded_coeffs = torch.zeros(15, device=device)
            padded_coeffs[:len(coeffs)] = coeffs
            coeffs = padded_coeffs
        elif len(coeffs) > 15:
            coeffs = coeffs[:15]

        phase_tensor = zernike_phase_from_coeffs(coeffs, grid_size=grid_size_high_res, device=device)
        E0_high_res = generate_vortex_beam(grid_size=grid_size_high_res, l=l, L=L, w=w, device=device)
        E_aberr_high_res = E0_high_res * torch.exp(1j * phase_tensor)
        propagated_field_high_res = angular_spectrum_propagation(E_aberr_high_res, wavelength=wavelength, z=z, L=L, device=device)
        intensity_high_res = torch.abs(propagated_field_high_res) ** 2
        
        # 光场图下采样
        intensity_high_res_np = intensity_high_res.cpu().numpy()
        intensity_low_res_np = cv2.resize(intensity_high_res_np, (grid_size_low_res, grid_size_low_res), interpolation=cv2.INTER_AREA)
        intensity_normalized = ((intensity_low_res_np - intensity_low_res_np.min()) / (intensity_low_res_np.max() - intensity_low_res_np.min()) * 255).astype(np.uint8)

        output_filename = f"sim_error_intensity_{file_num}.png"
        output_path = os.path.join("ConvertNet/datasets/sim_intensity", output_filename)
        cv2.imwrite(output_path, intensity_normalized)

if __name__ == "__main__":
    simulate_intensity_from_zernike_coeffs()



# import torch
# import cv2
# import os
# import numpy as np
# from tqdm import tqdm
# from lg_vortex import generate_vortex_beam
# from propagation import angular_spectrum_propagation

# def simulate_intensity_from_phase():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     os.makedirs("ConvertNet/datasets/sim_intensity", exist_ok=True)
#     phase_dir = "ConvertNet/datasets/phase"
#     phase_files = sorted([f for f in os.listdir(phase_dir) if f.startswith('ds_error_phase_') and f.endswith('.png')])

#     grid_size_high_res = 2000
#     grid_size_low_res = 400
#     wavelength = 532e-9
#     z = 300e-3
#     L = 6e-3
#     w = 2e-3
#     l = 20
    
#     for phase_file in tqdm(phase_files, desc="Simulating intensity"):
#         file_num = phase_file.split('_')[-1].split('.')[0]
#         phase_path = os.path.join(phase_dir, phase_file)
#         phase_img = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
        
#         if phase_img is None:
#             print(f"Warning: Could not read {phase_path}")
#             continue
#         if phase_img.shape != (grid_size_high_res, grid_size_high_res):
#             print(f"Warning: Phase image size {phase_img.shape} doesn't match expected size")
#             continue

#         phase_tensor = torch.from_numpy(phase_img.astype('float32')).to(device)
#         phase_tensor = phase_tensor / 255.0 * (2 * torch.pi)

#         E0_high_res = generate_vortex_beam(grid_size=grid_size_high_res, l=l, L=L, w=w, device=device)
#         E_aberr_high_res = E0_high_res * torch.exp(1j * phase_tensor)
#         propagated_field_high_res = angular_spectrum_propagation(E_aberr_high_res, wavelength=wavelength, z=z, L=L, device=device)
#         intensity_high_res = torch.abs(propagated_field_high_res) ** 2
        
#         # 光场图下采样
#         intensity_high_res_np = intensity_high_res.cpu().numpy()
#         intensity_low_res_np = cv2.resize(
#             intensity_high_res_np, 
#             (grid_size_low_res, grid_size_low_res), 
#             interpolation=cv2.INTER_AREA
#         )

#         intensity_normalized = ((intensity_low_res_np - intensity_low_res_np.min()) / (intensity_low_res_np.max() - intensity_low_res_np.min()) * 255).astype(np.uint8)
#         output_filename = f"sim_error_intensity_{file_num}.png"
#         output_path = os.path.join("ConvertNet/datasets/sim_intensity", output_filename)
#         cv2.imwrite(output_path, intensity_normalized)

# if __name__ == "__main__":
#     simulate_intensity_from_phase()