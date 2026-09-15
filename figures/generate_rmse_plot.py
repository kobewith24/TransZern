import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.font_manager as fm

# Use Arial for all text to match paper figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Data from Joint/results/rmse_per_mode.txt
modes = ['Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8',
         'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14', 'Z15']

before = np.array([0.582998, 0.559814, 0.585051, 0.579081, 0.560913,
                   0.571441, 0.547922, 0.556632, 0.563943, 0.581532,
                   0.579935, 0.569776, 0.595565, 0.561718])

after = np.array([0.089770, 0.073027, 0.071666, 0.049031, 0.060328,
                  0.032012, 0.022485, 0.020065, 0.027955, 0.082423,
                  0.044895, 0.029089, 0.048209, 0.078547])

x = np.arange(len(modes))
width = 0.35
offset = 0.22

# Broken axis: map actual values to evenly spaced display positions
# Display ticks: 0, 0.05, 0.10, 0.15, [break], 0.50, 0.55, 0.60, 0.65
# Each displayed tick interval has equal visual spacing.
tick_values = np.array([0, 0.05, 0.10, 0.15, 0.50, 0.55, 0.60, 0.65, 0.70])
break_idx = 3  # break between index 3 (0.15) and index 4 (0.50)


def build_transform(tick_values):
    """Map actual y to display y where each tick interval has equal height."""
    tick_values = np.asarray(tick_values)
    n_segments = len(tick_values) - 1
    # height of each visual segment
    segment_height = 1.0 / n_segments
    # break between index 3 (0.15) and index 4 (0.50)
    break_idx = 3

    def transform(y):
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        for i in range(n_segments):
            if i < break_idx:
                mask = (y >= tick_values[i]) & (y <= tick_values[i + 1])
                out[mask] = i * segment_height + \
                    (y[mask] - tick_values[i]) / (tick_values[i + 1] - tick_values[i]) * segment_height
            elif i == break_idx:
                # broken segment: do not map any real data here
                mask = (y > tick_values[i]) & (y < tick_values[i + 1])
                out[mask] = np.nan
            else:
                mask = (y >= tick_values[i]) & (y <= tick_values[i + 1])
                # skip one segment height for the break
                out[mask] = (i + 1) * segment_height + \
                    (y[mask] - tick_values[i]) / (tick_values[i + 1] - tick_values[i]) * segment_height
        return out

    def inv(pos):
        """Map display y position back to tick label value."""
        pos = np.asarray(pos, dtype=float)
        out = np.empty_like(pos)
        segment_height = 1.0 / n_segments
        for i in range(n_segments):
            if i < break_idx:
                lo, hi = i * segment_height, (i + 1) * segment_height
                mask = (pos >= lo) & (pos <= hi)
                out[mask] = tick_values[i] + (pos[mask] - lo) / segment_height * (tick_values[i + 1] - tick_values[i])
            elif i == break_idx:
                continue
            else:
                lo, hi = (i + 1) * segment_height, (i + 2) * segment_height
                mask = (pos >= lo) & (pos <= hi)
                out[mask] = tick_values[i] + (pos[mask] - lo) / segment_height * (tick_values[i + 1] - tick_values[i])
        return out

    return transform, inv


transform_y, inv_y = build_transform(tick_values)

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

# Y-tick positions: evenly spaced display positions for each tick
yticks_pos = np.linspace(0, 1, len(tick_values))
ax.set_yticks(yticks_pos[:-1].tolist() + [yticks_pos[-1]])
ax.set_yticklabels([f'{v:.2f}' for v in tick_values], fontsize=9)
ax.set_ylim(0, 1.0)

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
