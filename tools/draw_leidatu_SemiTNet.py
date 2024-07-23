import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 数据
data = {
    'Method': ['Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'Ours'],
    'IoU': [90.86, 90.50, 91.98, 93.23, 94.09, 94.32],
    'Dice': [92.83, 93.13, 94.36, 94.81, 95.64, 95.85],
    'Precision': [91.63, 87.84, 91.24, 93.57, 94.06, 94.84],
    'Recall': [95.66, 93.60, 96.20, 96.78, 97.30, 97.81],
    'F1': [93.60, 90.63, 93.66, 95.15, 95.65, 96.30]

}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置雷达图的标签
labels = df.columns[1:]
num_vars = len(labels)

# 计算角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(polar=True))

# 颜色列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 绘制每个方法的雷达图数据
for i in range(len(df)):
    values = df.iloc[i].drop('Method').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=df.iloc[i]['Method'])
    ax.fill(angles, values, color=colors[i], alpha=0.25)  # 设置alpha使填充区域半透明

# 调整范围，使得原点不是从0开始
ax.set_ylim(85, 98.5)

# 设置刻度标签
ax.set_yticks([85, 90, 95, 98.5])
ax.set_yticklabels(['85%', '90%', '95%', '98.5%'], fontsize=12)  # 增大 y 轴标签的字体大小

# 添加标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=15, fontweight='bold')

# 调整标签的角度
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

# 设置雷达图的0度起始位置
ax.set_theta_zero_location('N')

# 添加网格线
ax.grid(color='gray', linestyle='--', linewidth=1)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 添加图例
patches = [mpatches.Patch(facecolor=colors[i], edgecolor=colors[i], label=df.iloc[i]['Method'], lw=2) for i in range(len(df))]
ax.legend(handles=patches, labelspacing=0.1, fontsize='large')

# 添加标题
# plt.title('Segmentation and Identification Performance', size=20, color='#333333')

plt.savefig('a.png')
