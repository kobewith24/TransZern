import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from model import ConvertNet
from train import RealToSimDataset


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join("ConvertNet/models", "best_u.pth")
    model = ConvertNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    os.makedirs("ConvertNet/result", exist_ok=True)
    full_dataset = RealToSimDataset()
    test_dataset = Subset(full_dataset, range(900, 1000))    # 1800 ~ 2000
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    mse_list = []
    for idx, (real_tensor, sim_tensor) in enumerate(test_loader):
        real_tensor = real_tensor.to(device)
        sim_tensor = sim_tensor.to(device)
        with torch.no_grad():
            pred_tensor = model(real_tensor)
        mse = nn.MSELoss()(pred_tensor, sim_tensor).item()
        mse_list.append(mse)

        if idx % 40 == 0:
            print(f"Sample {idx}: MSE = {mse:.6f}")
            save_result_image(real_tensor, sim_tensor, pred_tensor, idx)

    overall_mse = np.mean(mse_list)
    print(f"Test Mean MSE(Test Loss) = {overall_mse:.6f}")

    with open(os.path.join("ConvertNet/result", "real2sim_mse.txt"), "w") as f:
        f.write(f"Test sample size: {len(mse_list)}\n")
        f.write(f"Test Mean MSE: {overall_mse:.6f}\n")
        f.write("\nTest sample MSE:\n")
        for i, mse in enumerate(mse_list):
            f.write(f"Sample {i}: {mse:.6f}\n")
    return overall_mse

def save_result_image(real_tensor, sim_tensor, pred_tensor, sample_idx):
    real_img = real_tensor.squeeze().cpu().numpy()
    sim_img = sim_tensor.squeeze().cpu().numpy()
    pred_img = pred_tensor.squeeze().cpu().numpy()
    # mse_value = np.mean((pred_img - sim_img) ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(real_img, cmap='gray')
    axes[0].set_title('Real distorted light field', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(sim_img, cmap='gray')
    axes[1].set_title('Simulated distorted light field', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(pred_img, cmap='gray')
    axes[2].set_title('Predicted-simulated distorted light field', fontsize=14)
    axes[2].axis('off')
    plt.tight_layout()

    existing = [f for f in os.listdir('ConvertNet/result') if f.startswith('lg_real2sim_') and f.endswith('.png')]
    nums = [int(f[len('lg_real2sim_'):-len('.png')]) for f in existing if f[len('lg_real2sim_'):-len('.png')].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    filename = f"ConvertNet/result/lg_real2sim_{next_num}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_mse = test()

# tensorboard --logdir=ConvertNet/logs_u --port=6006