import os
import numpy as np
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from dataset import generate_training_pair


def create_directory_structure():
    os.makedirs("ZernikeNet/datasets", exist_ok=True)
    os.makedirs("ZernikeNet/datasets/img_simulated", exist_ok=True)
    os.makedirs("ZernikeNet/datasets/csv_simulated", exist_ok=True)

def save_image_as_png(image_tensor, filepath):
    image_np = image_tensor.cpu().numpy()
    image_normalized = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
    cv2.imwrite(filepath, image_normalized)

def save_coefficients_as_csv(coeffs_tensor, filepath):
    coeffs_np = coeffs_tensor.cpu().numpy()
    df = pd.DataFrame(coeffs_np.reshape(1, -1))
    df.to_csv(filepath, index=False, header=False)

def generate_dataset(num_samples=10000, grid_size=400, l=20, num_modes=15, L=6e-3, w=2e-3):    # noqa: E741
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    create_directory_structure()
    
    # 保存理想光强图像
    I_distorted, I_ideal, coeffs, phi_aberr, E0, E_aberr = generate_training_pair(grid_size, l, num_modes, L, w, device=device)
    ideal_img_path = "ZernikeNet/datasets/target_intensity.bmp"
    save_image_as_png(I_ideal, ideal_img_path)
    
    # 生成训练数据集
    for i in tqdm(range(num_samples), desc="Generating dataset"):
        I_distorted, I_ideal, coeffs, phi_aberr, E0, E_aberr = generate_training_pair(grid_size, l, num_modes, L, w, device=device)
        
        distorted_img_path = f"ZernikeNet/datasets/img_simulated/{i}.png"
        save_image_as_png(I_distorted, distorted_img_path)
        
        coeffs_csv_path = f"ZernikeNet/datasets/csv_simulated/{i}.csv"
        save_coefficients_as_csv(coeffs, coeffs_csv_path)

if __name__ == "__main__":
    num_samples = 10000
    grid_size = 400
    l = 20    # noqa: E741
    num_modes = 15
    L = 6e-3
    w = 2e-3

    generate_dataset(num_samples, grid_size, l, num_modes, L, w)
    print("Done.")