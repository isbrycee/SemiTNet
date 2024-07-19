import torch
import copy
# 加载模型权重文件
model_weights = torch.load("/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/pretrained_model/SemiTNet_best_box_ap50.pth")

# 查看权重文件中的 keys
print("Original keys:")
for key in model_weights.keys():
    print(key)
# print(model_weights['model'].keys())
# 创建一个新的字典来存储修改后的权重
new_model_weights = {}

dicct = copy.deepcopy(model_weights['model'])

for key, value in dicct.items():
    new_key = key.replace('modelStudent', 'modelTeacher')
    model_weights['model'][new_key] = value
    del model_weights['model'][key]

# 查看修改后的 keys
print("Modified keys:")
for key in model_weights.keys():
    print(key)

# 保存修改后的模型权重文件
torch.save(model_weights, '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/pretrained_model/SemiTNet_best_box_ap50_convert.pth')

