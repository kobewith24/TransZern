import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 数据
sample_sizes = [100, 300, 600, 1000]
zr_values = [0.121, 0.077, 0.058, 0.052]
ir_values = [0.129, 0.106, 0.085, 0.081]

# 创建图形
fig, ax1 = plt.subplots(figsize=(5.0, 3.6))

# 绘制 ZR 折线（深蓝）- 点稍微小一点
ax1.plot(sample_sizes, zr_values, 'o-', color='#1f4e79', linewidth=2, markersize=6, label='Zernike RMSE (ZR)', zorder=3)

# 创建第二个 y 轴用于 IR - 点稍微小一点
ax2 = ax1.twinx()
ax2.plot(sample_sizes, ir_values, 's-', color='#5b9bd5', linewidth=2, markersize=6, label='Intensity RMSE (IR)', zorder=3)

# 设置坐标轴标签 (非加粗 Arial)
ax1.set_xlabel('Number of Experimental Samples', fontsize=6.5)
ax1.set_ylabel('Zernike RMSE (ZR)', fontsize=6.5, color='black')
ax2.set_ylabel('Intensity RMSE (IR)', fontsize=6.5, color='black')

# 设置 x 轴刻度
ax1.set_xticks(sample_sizes)
ax1.set_xticklabels(['100', '300', '600', '1000'], fontsize=5.5)

# 设置 y 轴范围
ax1.set_ylim(0.03, 0.15)
ax2.set_ylim(0.03, 0.15)

# 设置 y 轴刻度
ax1.set_yticks(np.arange(0.03, 0.16, 0.03))
ax2.set_yticks(np.arange(0.03, 0.16, 0.03))

# 设置刻度颜色和大小
ax1.tick_params(axis='y', labelcolor='black', labelsize=5.5)
ax2.tick_params(axis='y', labelcolor='black', labelsize=5.5)
ax1.tick_params(axis='x', labelsize=5.5)

# 添加网格
ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax1.set_axisbelow(True)

# 添加数值标签 - 离点远一点
for i, (x, zr, ir) in enumerate(zip(sample_sizes, zr_values, ir_values)):
    ax1.text(x, zr - 0.009, f'{zr:.3f}', ha='center', va='top', fontsize=5.5, color='#1f4e79', fontweight='bold')
    ax2.text(x, ir + 0.009, f'{ir:.3f}', ha='center', va='bottom', fontsize=5.5, color='#5b9bd5', fontweight='bold')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=6, framealpha=0.9)

plt.tight_layout()
plt.savefig('data_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('data_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Figure saved: data_sensitivity.png and data_sensitivity.pdf")
