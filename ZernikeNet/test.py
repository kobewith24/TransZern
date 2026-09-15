import numpy as np
import torch.nn as nn
import torch
from model import ZernikeNet
from dataset import PreGeneratedDataset
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from zernike import zernike_phase_from_coeffs
from propagation import angular_spectrum_propagation
from lg_vortex import generate_vortex_beam


def test():
    grid_size = 400
    num_modes = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载PINN训练后的模型
    model_path = os.path.join("ZernikeNet/models", "best_z.pth")
    model = ZernikeNet(num_modes=num_modes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 用测试集中的仿真样本进行测试 (索引9000 - 10000)
    full_dataset = PreGeneratedDataset(dataset_dir="ZernikeNet/datasets", device=device)
    test_dataset = Subset(full_dataset, range(9000, 10000))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    mse_list = []
    
    os.makedirs('ZernikeNet/result', exist_ok=True)

    coeffs_file_path = os.path.join('ZernikeNet/result', 'coeffs_simulated.txt')
    with open(coeffs_file_path, 'w') as f:
        f.write("{:<8}\t{:<15}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
            "Sample", "Type", "MSE", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12", "Z13", "Z14"))
    
    # PINN损失函数用于测试
    class TestPhysicalInformedLoss(nn.Module):
        def __init__(self, coeff_weight=1.0, intensity_weight=0.3, epsilon=1e-8):
            super().__init__()
            self.coeff_weight = coeff_weight
            self.intensity_weight = intensity_weight
            self.epsilon = epsilon
            self.register_buffer('ranges', torch.tensor([
                15.0, 15.0, 12.0, 12.0, 12.0,
                9.0, 9.0, 9.0, 9.0,
                6.0, 6.0, 6.0, 6.0, 6.0
            ]))
            
        def forward(self, coeffs_pred, coeffs_true, I_distorted, device):
            ranges = self.ranges.to(coeffs_pred.device)
            pred_norm = coeffs_pred[:, 1:15] / ranges
            true_norm = coeffs_true[:, 1:15] / ranges
            coeff_loss = nn.MSELoss()(pred_norm, true_norm)
            intensity_loss = self.compute_intensity_consistency_loss(
                coeffs_pred, coeffs_true, I_distorted, device
            )
            total_loss = (self.coeff_weight * coeff_loss + self.intensity_weight * intensity_loss)
            return total_loss
        
        def compute_intensity_consistency_loss(self, coeffs_pred, coeffs_true, I_distorted, device):
            batch_size = coeffs_pred.size(0)
            intensity_losses = []
            
            grid_size = 400
            l = 20
            L = 6e-3
            w = 2e-3
            wavelength = 532e-9
            z = 300e-3
            
            for i in range(batch_size):
                E0 = generate_vortex_beam(grid_size, l, L, w, device=device)
                phi_aberr_true = zernike_phase_from_coeffs(
                    coeffs_true[i], grid_size=grid_size, device=device
                )
                E_aberr = E0 * torch.exp(1j * phi_aberr_true)
                phi_correct_pred = zernike_phase_from_coeffs(
                    coeffs_pred[i], grid_size=grid_size, device=device
                )
                E_corrected = E_aberr * torch.exp(-1j * phi_correct_pred)
                U_corrected = angular_spectrum_propagation(
                    E_corrected, wavelength=wavelength, z=z, L=L, device=device
                )
                I_corrected_pred = torch.abs(U_corrected) ** 2
                
                U_ideal = angular_spectrum_propagation(
                    E0, wavelength=wavelength, z=z, L=L, device=device
                )
                I_ideal = torch.abs(U_ideal) ** 2
                
                I_pred_norm = (I_corrected_pred - I_corrected_pred.min()) / (I_corrected_pred.max() - I_corrected_pred.min() + self.epsilon)
                I_ideal_norm = (I_ideal - I_ideal.min()) / (I_ideal.max() - I_ideal.min() + self.epsilon)
                
                loss = nn.MSELoss()(I_pred_norm, I_ideal_norm)
                intensity_losses.append(loss)
            
            return torch.mean(torch.stack(intensity_losses))
    
    test_loss_fn = TestPhysicalInformedLoss(coeff_weight=1.0, intensity_weight=0.3)
    
    for idx, (I_tensor, coeffs_true) in enumerate(test_loader):
        I_tensor = I_tensor.to(device)
        coeffs_true = coeffs_true.to(device)

        with torch.no_grad():
            coeffs_pred = model(I_tensor)
            
            # 计算PINN损失
            total_loss = test_loss_fn(coeffs_pred, coeffs_true, I_tensor, device)
            
            mse_list.append(total_loss.item())
            
            coeffs_pred_np = coeffs_pred.cpu().numpy()[0]
            coeffs_true_np = coeffs_true.cpu().numpy()[0]
        
        # 每 100 张图输出一次结果
        if idx % 100 == 0:
            print(f"Sample {idx}: MSE = {total_loss.item():.6f}")
            print(f"Predicted Zernike coefficients: {[f'{x:.3f}' for x in coeffs_pred_np[1:15]]}")
            print(f"True Zernike coefficients: {[f'{x:.3f}' for x in coeffs_true_np[1:15]]}\n")

        abs_errors = np.abs(coeffs_pred_np[1:15] - coeffs_true_np[1:15])
        with open(coeffs_file_path, 'a') as f:
            pred_values = [f"{x:.6f}" for x in coeffs_pred_np[:15]]
            f.write("{:<8}\t{:<15}\t{:<12.6f}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
                idx, "Predicted", total_loss.item(), *pred_values))
            true_values = [f"{x:.6f}" for x in coeffs_true_np[:15]]
            f.write("{:<8}\t{:<15}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
                idx, "True", "", *true_values))
            error_values = [f"{x:.6f}" for x in abs_errors]
            all_error_values = [""] + error_values
            f.write("{:<8}\t{:<15}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
                idx, "Absolute Error", "", *all_error_values))

        if idx % 100 == 0:
            # 可视化校正效果
            l = 20
            L = 6e-3
            w = 2e-3
            wavelength = 532e-9
            z = 300e-3

            E0 = generate_vortex_beam(grid_size, l, L, w, device=device)
            phi_aberr = zernike_phase_from_coeffs(torch.from_numpy(coeffs_true_np).to(device), grid_size=2000, device=device)
            E0_highres = generate_vortex_beam(2000, l, L, w, device=device)
            E_aberr_highres = E0_highres * torch.exp(1j * phi_aberr)
            phi_aberr_lowres = zernike_phase_from_coeffs(torch.from_numpy(coeffs_true_np).to(device), grid_size=grid_size, device=device)
            E_aberr_lowres = E0 * torch.exp(1j * phi_aberr_lowres)
            U_distorted = angular_spectrum_propagation(E_aberr_lowres, wavelength=wavelength, z=z, L=L, device=device)
            I_distorted = torch.abs(U_distorted) ** 2

            phi_pred_aberr = zernike_phase_from_coeffs(torch.from_numpy(coeffs_pred_np).to(device), grid_size=2000, device=device)
            E_corrected_highres = E_aberr_highres * torch.exp(-1j * phi_pred_aberr)
            phi_pred_aberr_lowres = zernike_phase_from_coeffs(torch.from_numpy(coeffs_pred_np).to(device), grid_size=grid_size, device=device)
            E_corrected_lowres = E_aberr_lowres * torch.exp(-1j * phi_pred_aberr_lowres)
            U_corrected = angular_spectrum_propagation(E_corrected_lowres, wavelength=wavelength, z=z, L=L, device=device)
            I_corrected = torch.abs(U_corrected) ** 2

            U_ideal = angular_spectrum_propagation(E0, wavelength=wavelength, z=z, L=L, device=device)
            I_ideal = torch.abs(U_ideal) ** 2

            mse_distorted = torch.mean((I_distorted - I_ideal) ** 2).item()
            mse_corrected = torch.mean((I_corrected - I_ideal) ** 2).item()

            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
            axs[0, 0].imshow(I_distorted.cpu().numpy(), cmap='gray')
            axs[0, 0].set_title('Simulated distorted light field', fontsize=15)
            axs[0, 0].text(
                0.5, -0.03,
                f"MSE: {mse_distorted:.4e}",
                ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=15
            )
            
            axs[0, 1].imshow(torch.angle(E_aberr_highres).cpu().numpy(), cmap='gray')
            axs[0, 1].set_title('Distorted phase', fontsize=15)
            
            axs[0, 2].imshow(phi_aberr.cpu().numpy(), cmap='gray')
            axs[0, 2].set_title('Aberration phase', fontsize=15)

            axs[1, 0].imshow(I_corrected.cpu().numpy(), cmap='gray')
            axs[1, 0].set_title('Simulated corrected light field', fontsize=15)
            axs[1, 0].text(
                0.5, -0.03,
                f"MSE: {mse_corrected:.4e}",
                ha='center', va='top', transform=axs[1, 0].transAxes, fontsize=15
            )
            
            axs[1, 1].imshow(torch.angle(E_corrected_highres).cpu().numpy(), cmap='gray')
            axs[1, 1].set_title('Corrected phase', fontsize=15)
            
            axs[1, 2].imshow(phi_pred_aberr.cpu().numpy(), cmap='gray')
            axs[1, 2].set_title('Predicted aberration phase', fontsize=15)
            
            for ax in axs.flat:
                ax.axis('off')

            plt.tight_layout(pad=3.0)

            existing = [f for f in os.listdir('ZernikeNet/result') if f.startswith('lg_simulated_') and f.endswith('.png')]
            nums = [int(f[len('lg_simulated_'):-len('.png')]) for f in existing if f[len('lg_simulated_'):-len('.png')].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            save_path = os.path.join('ZernikeNet/result', f'lg_simulated_{next_num}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    overall_mse = np.mean(mse_list)
    print(f"\nTest Mean MSE(Test Loss) = {overall_mse:.6f}.")
    return overall_mse


if __name__ == "__main__":
    test_mse = test()

# tensorboard --logdir=ZernikeNet/logs_z --port=6006




# import numpy as np
# import torch.nn as nn
# import torch
# from model import ZernikeNet
# from dataset import PreGeneratedDataset
# import matplotlib.pyplot as plt
# import os
# from torch.utils.data import DataLoader, Subset
# from zernike import zernike_phase_from_coeffs
# from propagation import angular_spectrum_propagation
# from lg_vortex import generate_vortex_beam


# def test():
#     grid_size = 400
#     num_modes = 15
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 加载PINN训练后的模型
#     model_path = os.path.join("ZernikeNet/models", "best_z.pth")
#     model = ZernikeNet(num_modes=num_modes).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model.eval()
    
#     # 用测试集中的仿真样本进行测试 (索引4500 - 4999)
#     full_dataset = PreGeneratedDataset(dataset_dir="ZernikeNet/datasets", device=device)
#     test_dataset = Subset(full_dataset, range(4500, 5000))
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
#     mse_list = []
#     image_mse_list = []  # 新增：用于单独存储图像MSE的列表
    
#     os.makedirs('ZernikeNet/result', exist_ok=True)

#     coeffs_file_path = os.path.join('ZernikeNet/result', 'coeffs_simulated.txt')
#     with open(coeffs_file_path, 'w') as f:
#         f.write("{:<8}\t{:<15}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
#             "Sample", "Type", "Total Loss", "Img_MSE", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12", "Z13", "Z14"))
    
#     # PINN损失函数用于测试
#     class TestPhysicalInformedLoss(nn.Module):
#         def __init__(self, coeff_weight=1.0, intensity_weight=0.3, epsilon=1e-8):
#             super().__init__()
#             self.coeff_weight = coeff_weight
#             self.intensity_weight = intensity_weight
#             self.epsilon = epsilon
            
#         def forward(self, coeffs_pred, coeffs_true, I_distorted, device):
#             coeff_loss = nn.MSELoss()(coeffs_pred[:, 1:15], coeffs_true[:, 1:15])
#             intensity_loss = self.compute_intensity_consistency_loss(
#                 coeffs_pred, coeffs_true, I_distorted, device
#             )
#             total_loss = (self.coeff_weight * coeff_loss + self.intensity_weight * intensity_loss)
#             return total_loss, coeff_loss, intensity_loss  # 修改：返回多个值
        
#         def compute_intensity_consistency_loss(self, coeffs_pred, coeffs_true, I_distorted, device):
#             batch_size = coeffs_pred.size(0)
#             intensity_losses = []
            
#             grid_size = 400
#             l = 20
#             L = 6e-3
#             w = 2e-3
#             wavelength = 532e-9
#             z = 300e-3
            
#             for i in range(batch_size):
#                 E0 = generate_vortex_beam(grid_size, l, L, w, device=device)
#                 phi_aberr_true = zernike_phase_from_coeffs(
#                     coeffs_true[i], grid_size=grid_size, device=device
#                 )
#                 E_aberr = E0 * torch.exp(1j * phi_aberr_true)
#                 phi_correct_pred = zernike_phase_from_coeffs(
#                     coeffs_pred[i], grid_size=grid_size, device=device
#                 )
#                 E_corrected = E_aberr * torch.exp(-1j * phi_correct_pred)
#                 U_corrected = angular_spectrum_propagation(
#                     E_corrected, wavelength=wavelength, z=z, L=L, device=device
#                 )
#                 I_corrected_pred = torch.abs(U_corrected) ** 2
                
#                 U_ideal = angular_spectrum_propagation(
#                     E0, wavelength=wavelength, z=z, L=L, device=device
#                 )
#                 I_ideal = torch.abs(U_ideal) ** 2
                
#                 I_pred_norm = (I_corrected_pred - I_corrected_pred.min()) / (I_corrected_pred.max() - I_corrected_pred.min() + self.epsilon)
#                 I_ideal_norm = (I_ideal - I_ideal.min()) / (I_ideal.max() - I_ideal.min() + self.epsilon)
                
#                 loss = nn.MSELoss()(I_pred_norm, I_ideal_norm)
#                 intensity_losses.append(loss)
            
#             return torch.mean(torch.stack(intensity_losses))
    
#     test_loss_fn = TestPhysicalInformedLoss(coeff_weight=1.0, intensity_weight=0.3)
    
#     for idx, (I_tensor, coeffs_true) in enumerate(test_loader):
#         I_tensor = I_tensor.to(device)
#         coeffs_true = coeffs_true.to(device)

#         input_val = I_tensor.repeat(1, 2, 1, 1).to(device)
        
#         with torch.no_grad():
#             coeffs_pred = model(input_val)
            
#             # 计算PINN损失，现在返回三个值
#             total_loss, coeff_loss, intensity_loss = test_loss_fn(coeffs_pred, coeffs_true, I_tensor, device)
            
#             mse_list.append(total_loss.item())
#             image_mse_list.append(intensity_loss.item())  # 记录图像MSE
            
#             coeffs_pred_np = coeffs_pred.cpu().numpy()[0]
#             coeffs_true_np = coeffs_true.cpu().numpy()[0]
        
#         # 每 100 张图输出一次结果
#         if idx % 100 == 0:
#             print(f"Sample {idx}: Total Loss = {total_loss.item():.6f}, Image MSE = {intensity_loss.item():.6f}")
#             print(f"Predicted Zernike coefficients: {[f'{x:.3f}' for x in coeffs_pred_np[1:15]]}")
#             print(f"True Zernike coefficients: {[f'{x:.3f}' for x in coeffs_true_np[1:15]]}\n")

#         abs_errors = np.abs(coeffs_pred_np[1:15] - coeffs_true_np[1:15])
#         with open(coeffs_file_path, 'a') as f:
#             pred_values = [f"{x:.6f}" for x in coeffs_pred_np[:15]]
#             f.write("{:<8}\t{:<15}\t{:<12.6f}\t{:<12.6f}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
#                 idx, "Predicted", total_loss.item(), intensity_loss.item(), *pred_values))
#             true_values = [f"{x:.6f}" for x in coeffs_true_np[:15]]
#             f.write("{:<8}\t{:<15}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
#                 idx, "True", "", "", *true_values))
#             error_values = [f"{x:.6f}" for x in abs_errors]
#             all_error_values = ["", ""] + error_values
#             f.write("{:<8}\t{:<15}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
#                 idx, "Absolute Error", "", "", *all_error_values))

#         if idx % 100 == 0:
#             # 可视化校正效果
#             l = 20
#             L = 6e-3
#             w = 2e-3
#             wavelength = 532e-9
#             z = 300e-3

#             E0 = generate_vortex_beam(grid_size, l, L, w, device=device)
#             phi_aberr = zernike_phase_from_coeffs(torch.from_numpy(coeffs_true_np).to(device), grid_size=2000, device=device)
#             E0_highres = generate_vortex_beam(2000, l, L, w, device=device)
#             E_aberr_highres = E0_highres * torch.exp(1j * phi_aberr)
#             phi_aberr_lowres = zernike_phase_from_coeffs(torch.from_numpy(coeffs_true_np).to(device), grid_size=grid_size, device=device)
#             E_aberr_lowres = E0 * torch.exp(1j * phi_aberr_lowres)
#             U_distorted = angular_spectrum_propagation(E_aberr_lowres, wavelength=wavelength, z=z, L=L, device=device)
#             I_distorted = torch.abs(U_distorted) ** 2

#             phi_pred_aberr = zernike_phase_from_coeffs(torch.from_numpy(coeffs_pred_np).to(device), grid_size=2000, device=device)
#             E_corrected_highres = E_aberr_highres * torch.exp(-1j * phi_pred_aberr)
#             phi_pred_aberr_lowres = zernike_phase_from_coeffs(torch.from_numpy(coeffs_pred_np).to(device), grid_size=grid_size, device=device)
#             E_corrected_lowres = E_aberr_lowres * torch.exp(-1j * phi_pred_aberr_lowres)
#             U_corrected = angular_spectrum_propagation(E_corrected_lowres, wavelength=wavelength, z=z, L=L, device=device)
#             I_corrected = torch.abs(U_corrected) ** 2

#             U_ideal = angular_spectrum_propagation(E0, wavelength=wavelength, z=z, L=L, device=device)
#             I_ideal = torch.abs(U_ideal) ** 2

#             mse_distorted = torch.mean((I_distorted - I_ideal) ** 2).item()
#             mse_corrected = torch.mean((I_corrected - I_ideal) ** 2).item()

#             fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
#             axs[0, 0].imshow(I_distorted.cpu().numpy(), cmap='gray')
#             axs[0, 0].set_title('Distorted Intensity', fontsize=15)
#             axs[0, 0].text(
#                 0.5, -0.03,
#                 f"MSE: {mse_distorted:.4e}",
#                 ha='center', va='top', transform=axs[0, 0].transAxes, fontsize=15
#             )
            
#             axs[0, 1].imshow(torch.angle(E_aberr_highres).cpu().numpy(), cmap='gray')
#             axs[0, 1].set_title('Distorted Phase', fontsize=15)
            
#             axs[0, 2].imshow(phi_aberr.cpu().numpy(), cmap='gray')
#             axs[0, 2].set_title('True Aberration Phase', fontsize=15)

#             axs[1, 0].imshow(I_corrected.cpu().numpy(), cmap='gray')
#             axs[1, 0].set_title('Corrected Intensity', fontsize=15)
#             axs[1, 0].text(
#                 0.5, -0.03,
#                 f"MSE: {mse_corrected:.4e}",
#                 ha='center', va='top', transform=axs[1, 0].transAxes, fontsize=15
#             )
            
#             axs[1, 1].imshow(torch.angle(E_corrected_highres).cpu().numpy(), cmap='gray')
#             axs[1, 1].set_title('Corrected Phase', fontsize=15)
            
#             axs[1, 2].imshow(phi_pred_aberr.cpu().numpy(), cmap='gray')
#             axs[1, 2].set_title('Predicted Aberration Phase', fontsize=15)
            
#             for ax in axs.flat:
#                 ax.axis('off')

#             plt.tight_layout(pad=3.0)

#             existing = [f for f in os.listdir('ZernikeNet/result') if f.startswith('lg_simulated_') and f.endswith('.png')]
#             nums = [int(f[len('lg_simulated_'):-len('.png')]) for f in existing if f[len('lg_simulated_'):-len('.png')].isdigit()]
#             next_num = max(nums) + 1 if nums else 1
#             save_path = os.path.join('ZernikeNet/result', f'lg_simulated_{next_num}.png')
#             fig.savefig(save_path, dpi=300, bbox_inches='tight')
#             plt.close(fig)

#     overall_mse = np.mean(mse_list)
#     overall_image_mse = np.mean(image_mse_list)  # 计算平均图像MSE
#     print(f"\nTest Mean MSE (Weighted Total Loss) = {overall_mse:.6f}.")
#     print(f"Test Mean MSE (Image Consistency Only) = {overall_image_mse:.6f}.")
#     return overall_mse, overall_image_mse  # 返回两个值


# if __name__ == "__main__":
#     test_mse, test_image_mse = test()  # 接收两个返回值