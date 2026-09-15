# 消融实验：Without ConvertNet

本目录包含消融实验代码，用于验证 ConvertNet (Domain Conversion) 的有效性。

## 实验设计

- **With ConvertNet**: 实验光强 → ConvertNet → 仿真光强 → ZernikeNet → Zernike系数 (原方法)
- **Without ConvertNet**: 实验光强 → ZernikeNet → Zernike系数 (消融方法)

## 文件说明

- `dataset.py`: 消融实验数据集，只加载实验光强和Zernike系数（跳过仿真光强）
- `train.py`: 训练脚本，直接使用实验数据训练 ZernikeNet
- `compare.py`: 对比脚本，同时评估有/无 ConvertNet 的效果
- `datasets/`: 符号链接到 Joint/datasets/（包含 real_intensity 和 csv）
- `results/`: 对比结果输出目录

## 使用步骤

### 1. 训练消融模型（直接训练 ZernikeNet）

```bash
cd /home/wangquan/Aberration_correction
python SimulatedDataOnly/train.py
```

训练参数：
- Epochs: 60
- Batch size: 4
- Learning rate: 1e-4 (cosine annealing)
- Optimizer: AdamW (weight_decay=1e-5)
- Loss: Zernike MSE + Regularization

模型将保存在 `SimulatedDataOnly/models/best_direct.pth`

### 2. 对比评估

确保 Joint 模型已训练完成（`Joint/models/best_joint.pth` 存在），然后运行：

```bash
cd /home/wangquan/Aberration_correction
python SimulatedDataOnly/compare.py
```

输出：
- `SimulatedDataOnly/results/ablation_comparison.csv`: 汇总对比表
- `SimulatedDataOnly/results/detailed_comparison.csv`: 每个样本的详细对比

## 预期结果

如果 ConvertNet 有效，预期结果应为：
- **With ConvertNet** 的 RMSE 显著低于 **Without ConvertNet**
- 改进幅度预计在 30-60% 左右（取决于 domain gap 的大小）

如果两者性能接近，说明：
1. 实验数据与仿真数据的 domain gap 较小，ConvertNet 不是关键组件
2. 或者 ZernikeNet 本身足够鲁棒，可以直接从实验光强中提取特征

## 注意事项

1. 训练消融模型时，使用的是与 Joint 训练相同的数据集划分（70% train, 15% val, 15% test）
2. 对比时，两个模型在完全相同的测试集上评估，确保公平性
3. 如果显存不足，可以在 `train.py` 中将 `batch_size` 调小（如改为 2）
