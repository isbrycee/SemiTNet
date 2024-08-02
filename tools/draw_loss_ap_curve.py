import re
import matplotlib.pyplot as plt

# Sample log data (trimmed for brevity)
with open('log_for_draw.txt', 'r') as f:
    log_data = f.read()

# Regular expressions to extract data
iter_pattern = re.compile(r'iter: (\d+)')
loss_pattern = re.compile(r'total_loss: ([\d.]+)')
# ap50_pattern = re.compile(r'AP50: ([\d.]+)')

iterations = []
losses = []
ap50_values = []

for line in log_data.splitlines():
    iter_match = iter_pattern.search(line)
    loss_match = loss_pattern.search(line)
    # ap50_match = ap50_pattern.search(line)
    
    if iter_match and loss_match:
        iterations.append(int(iter_match.group(1)))
        losses.append(float(loss_match.group(1)))
 
# Sanity check to ensure data is collected correctly
# print(iterations)
# print(losses)

with open('log_for_draw.txt', 'r') as f:
    log_data = f.readlines()
    for i, line in enumerate(log_data):
        if "d2.evaluation.testing INFO: copypaste: Task: segm" in line:
            if (i+2) < len(log_data):
                value = log_data[i+2].split('d2.evaluation.testing INFO: copypaste: ')[-1].split(',')[1]
                ap50_values.append(float(value))

# for drawing
print(len(iterations))

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax1 = plt.subplots()

# Plot total loss
color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(iterations, losses, color=color, label='Training Loss on Train Set')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for ap50
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Precision(%)', color=color)
ax2.plot(iterations[::25], ap50_values, color=color, label='Precision on Test Set')
ax2.tick_params(axis='y', labelcolor=color)

# Title and legends
plt.title('Loss and Precision vs. Iteration')
fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.85)) # loc='upper right'
plt.savefig('loss_ap50_vs_iteration.png')