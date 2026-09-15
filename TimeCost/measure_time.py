import torch
import numpy as np
import os
import time
import sys
import itertools
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Joint'))
from model import JointAberrationCorrection
from dataset import JointDataset


def measure_inference_time(device, model, test_loader, num_inferences, num_warmup=50):
    """Measure per-sample inference latency on a given device.

    The loader is cycled until num_inferences forward passes have been timed.
    """
    model.eval()

    # Warm-up
    with torch.no_grad():
        for i, (real_img, _, _) in enumerate(itertools.cycle(test_loader)):
            if i >= num_warmup:
                break
            real_img = real_img.to(device)
            _ = model(real_img)

    # Timed inference
    times = []
    with torch.no_grad():
        for i, (real_img, _, _) in enumerate(itertools.cycle(test_loader)):
            if i >= num_inferences:
                break
            real_img = real_img.to(device)
            # Synchronize before timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(real_img)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)  # ms

    times = np.array(times)
    return times


def main():
    data_dir = os.path.join('..', 'Joint', 'datasets')
    model_path = os.path.join('..', 'Joint', 'models', 'final_joint.pth')
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset (Joint 1300~1600, total 300 samples)
    full_dataset = JointDataset(data_dir=data_dir, device='cpu')
    test_dataset = Subset(full_dataset, range(1300, 1600))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Inference budget per device (plan.txt method 2: 1000 GPU inferences;
    # CPU is optional and kept shorter because CPU latency is ~4 s/frame)
    device_configs = []
    if torch.cuda.is_available():
        device_configs.append({
            'device': torch.device('cuda:0'),
            'num_inferences': 1000,
            'num_warmup': 50,
        })
    device_configs.append({
        'device': torch.device('cpu'),
        'num_inferences': 100,
        'num_warmup': 10,
    })

    results = {}
    for cfg in device_configs:
        device = cfg['device']
        num_inferences = cfg['num_inferences']
        num_warmup = cfg['num_warmup']

        print(f"\nMeasuring inference time on {device} "
              f"({num_warmup} warm-up + {num_inferences} timed) ...")
        model = JointAberrationCorrection().to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        times = measure_inference_time(device, model, test_loader,
                                       num_inferences=num_inferences,
                                       num_warmup=num_warmup)
        results[str(device)] = times

        print(f"  Inferences: {len(times)}")
        print(f"  Mean:   {np.mean(times):.3f} ms")
        print(f"  Std:    {np.std(times):.3f} ms")
        print(f"  Min:    {np.min(times):.3f} ms")
        print(f"  Max:    {np.max(times):.3f} ms")
        print(f"  Median: {np.median(times):.3f} ms")
        print(f"  FPS:    {1000.0 / np.median(times):.1f}")

    # Save results
    summary_path = os.path.join(output_dir, 'time_cost_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Device\tInferences\tMean(ms)\tStd(ms)\tMin(ms)\tMax(ms)\tMedian(ms)\tFPS\n")
        for device_name, times in results.items():
            f.write(f"{device_name}\t"
                    f"{len(times)}\t"
                    f"{np.mean(times):.3f}\t"
                    f"{np.std(times):.3f}\t"
                    f"{np.min(times):.3f}\t"
                    f"{np.max(times):.3f}\t"
                    f"{np.median(times):.3f}\t"
                    f"{1000.0 / np.median(times):.1f}\n")
    print(f"\nSummary saved to: {summary_path}")

    # Save raw times
    raw_path = os.path.join(output_dir, 'time_cost_raw.npy')
    np.save(raw_path, results)
    print(f"Raw times saved to: {raw_path}")


if __name__ == "__main__":
    main()
