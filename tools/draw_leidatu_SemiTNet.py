import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# edentulous
data = {
    'Method': ['Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'Ours'],
    'IoU': [89.65, 91.67, 92.24, 92.74, 92.35, 93.00],
    'Dice': [90.70, 93.09, 93.33, 94.31, 93.39, 94.30],
    'Precision': [90.44, 89.23, 92.28, 92.18, 92.46, 93.42],
    'Recall': [92.74, 92.51, 94.82, 95.61, 95.09, 96.40],
    'F1': [91.57, 90.84, 93.53, 93.86, 93.76, 94.89]
}

# dentate
data = {
    'Method': ['Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'Ours'],
    'IoU': [98.84, 99.24, 99.47, 99.53, 99.84, 99.76],
    'Dice': [98.98, 99.32, 99.59, 99.64, 99.86, 99.78],
    'Precision': [99.04, 97.64, 99.26, 99.45, 99.65, 99.69],
    'Recall': [99.36, 97.82, 99.57, 99.64, 99.65, 99.72],
    'F1': [99.20, 97.73, 99.41, 99.55, 99.65, 99.70]
}
# All test
data = {
    'Method': ['Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'Ours'],
    'IoU': [91.58, 93.26, 94.16, 93.75, 93.92, 94.41],
    'Dice': [92.44, 94.39, 95.43, 94.64, 94.75, 95.45],
    'Precision': [92.24, 90.99, 93.70, 93.74, 93.96, 94.74],
    'Recall': [94.13, 93.63, 96.45, 95.81, 96.04, 97.10],
    'F1': [93.17, 92.29, 95.06, 94.76, 94.99, 95.90]
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

## 调整范围，使得原点不是从0开始
ax.set_ylim(85, 97.5)

## 设置刻度标签
ax.set_yticks([85, 90, 95, 97.5])
ax.set_yticklabels(['90%', '90%', '95%', '97.5%'], fontsize=12)  # 增大 y 轴标签的字体大小

## 调整范围，使得原点不是从0开始
#ax.set_ylim(97, 100)

## 设置刻度标签
#ax.set_yticks([97, 98, 99, 100])
#ax.set_yticklabels(['97%', '98%', '99%', '100%'], fontsize=12)  # 增大 y 轴标签的字体大小

# 调整范围，使得原点不是从0开始
# ax.set_ylim(83, 97)

# 设置刻度标签
# ax.set_yticks([83, 89, 94, 97])
# ax.set_yticklabels(['83%', '89%', '94%', '97%'], fontsize=12)  # 增大 y 轴标签的字体大小

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
ax.legend(handles=patches, labelspacing=0.1, fontsize='large', loc='upper right')

# 添加标题
# plt.title('Segmentation and Identification Performance', size=20, color='#333333')

plt.savefig('SemiTNet_radar_all_test.svg', format="svg", dpi=300, bbox_inches='tight')
