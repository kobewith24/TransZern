import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use Arial for all text to match paper figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Display labels use one-based Noll indices.
modes = ['Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8',
         'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14', 'Z15']

before = np.array([0.582998, 0.559814, 0.585051, 0.579081, 0.560913,
                   0.571441, 0.547922, 0.556632, 0.563943, 0.581532,
                   0.579935, 0.569776, 0.595565, 0.561718])

after = np.array([0.089770, 0.073027, 0.071666, 0.049031, 0.060328,
                  0.032012, 0.022485, 0.020065, 0.027955, 0.082423,
                  0.044895, 0.029089, 0.048209, 0.078547])

# The arrays above follow the stored coefficient order (piston excluded).
# Convert to Noll order for display, preserving each mode's two measurements.
noll_order = np.array([2, 1, 4, 3, 5, 7, 8, 6, 9, 12, 13, 11, 14, 10]) - 1
before = before[noll_order]
after = after[noll_order]

x = np.arange(len(modes))
width = 0.35
offset = 0.22

# Keep each 0.05 interval uniformly spaced and reserve one interval for the break.
tick_values = np.array([0, 0.05, 0.10, 0.15, 0.45, 0.50, 0.55, 0.60, 0.65])
tick_positions = np.linspace(0, 1, len(tick_values))


def transform_y(values):
    """Map the displayed data ranges to the same coordinates as their ticks."""
    return np.interp(values, tick_values, tick_positions)


fig, ax = plt.subplots(figsize=(12, 5.5))

# Bars on transformed coordinates
rects1 = ax.bar(x - offset, transform_y(before), width, label='Before Correction',
                color='#D8BC9E')
rects2 = ax.bar(x + offset, transform_y(after), width, label='After Correction',
                color='#BFDBFE')

# Average horizontal lines (transformed)
mean_before = np.mean(before)
mean_after = np.mean(after)
ax.axhline(transform_y(mean_before), color='#DC2626', linestyle='--', linewidth=1.5,
           dashes=(8, 4), zorder=1)
ax.axhline(transform_y(mean_after), color='#16A34A', linestyle='--', linewidth=1.5,
           dashes=(8, 4), zorder=1)

ax.text(len(modes) / 2, transform_y(mean_before) + 0.015,
        f'Before Avg. RMSE = {mean_before:.3f}',
        color='#DC2626', fontsize=11, va='bottom', ha='center')
ax.text(len(modes) / 2, transform_y(mean_after) + 0.015,
        f'After Avg. RMSE = {mean_after:.3f}',
        color='#16A34A', fontsize=11, va='bottom', ha='center')

# Draw the break on the left spine between the two displayed ranges.
ax.set_yticks(tick_positions)
ax.set_yticklabels([f'{v:.2f}' for v in tick_values], fontsize=9)
ax.set_ylim(0, 1.0)
ax.spines['left'].set_visible(False)
break_low, break_high = tick_positions[3:5]
break_height = break_high - break_low
ax.plot([0, 0, 0.006, -0.006, 0.006, 0, 0],
        [0, break_low, break_low + 0.25 * break_height,
         (break_low + break_high) / 2, break_high - 0.25 * break_height,
         break_high, 1],
        color='black', linewidth=0.8, transform=ax.transAxes, clip_on=False)

# Custom xtick labels with important modes annotated on two lines
important = {2, 3, 4, 5, 6, 7, 8, 11}
mode_names = {
    'Z2': 'X-Tilt',
    'Z3': 'Y-Tilt',
    'Z4': 'Defocus',
    'Z5': 'Astig. 45°',
    'Z6': 'Astig. 0°',
    'Z7': 'Y-Coma',
    'Z8': 'X-Coma',
    'Z11': 'Spherical',
}

tick_labels = []
for i, m in enumerate(modes):
    n = i + 2
    if n in important and m in mode_names:
        tick_labels.append(f'{m}\n({mode_names[m]})')
    elif n in important:
        tick_labels.append(m)
    else:
        tick_labels.append(m)

ax.set_ylabel('Normalized RMSE', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('zernike_rmse_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved zernike_rmse_comparison.png")
