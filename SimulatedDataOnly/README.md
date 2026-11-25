# 消融实验：仅使用仿真训练

该基线使用仿真光强和对应的 Zernike 系数训练 ZernikeNet，并直接在实采光强上进行系数回归。对比脚本在同一组 Joint 测试样本上运行完整 TransZern 和仅仿真训练的基线。

## 使用步骤

在仓库根目录运行：

```bash
python SimulatedDataOnly/train_sim_only.py
python SimulatedDataOnly/compare.py
```

对比脚本读取 `Joint/models/final_joint.pth` 和 `SimulatedDataOnly/models/final_sim_only.pth`，并使用 `Joint/datasets` 中按排序后零基下标 1700–1999 选取的 300 个测试样本。

## 指标与输出

先汇总全部测试样本，计算每个非活塞模式的 RMSE，并除以该模式的系数半范围。论文使用这 14 个逐模式归一化 RMSE 的算术平均。脚本另行输出逐样本的指标。

- `results/ablation_comparison.csv`：汇总结果。
- `results/detailed_per_mode.csv`：按代码内部顺序给出的逐模式归一化 RMSE。
- `results/detailed_comparison.csv`：逐样本对比结果。
