import json
import numpy as np
from pycocotools import mask as maskUtils
import cv2
import os

train_json_path = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/datasets/MICCAI2024_SemiTeeth/Train-Labeled/train.json'
valid_json_path = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/datasets/MICCAI2024_SemiTeeth/validation.json'
inference_json_path = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/output/inference/coco_instances_results.json'
# inference_json_path = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/MICCAI2024_output_allResize_lr1e-4_bs16_pseudo_mask/train_1st_teacher_lr5e-5_iter5000/inference/coco_instances_results.json'
save_json_path = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/output/inference/prediction'
SCORE_THRESHOLD = 0.45

with open(train_json_path, 'r') as f:
    train_data = json.load(f)

category_id_to_fdi = {}
for cls_ in train_data['categories']:
    category_id_to_fdi[cls_['id']] = cls_['name']

imgID_2_width_height = {}
with open(valid_json_path, 'r') as f:
    valid_data = json.load(f)
    for img in valid_data['images']:
        imgID_2_width_height[img['id']] = (img['width'], img['height'], img['file_name'])

# 读取 coco_instances_results.json 文件
with open(inference_json_path, 'r') as f:
    coco_results = json.load(f)

image_results = {}

for item in coco_results:
    score = item["score"]
    if score < SCORE_THRESHOLD:
        continue
    image_id = item["image_id"]
    category_id = item["category_id"]
    segmentation = item["segmentation"]
    size = segmentation["size"]
    counts = segmentation["counts"]
    height = imgID_2_width_height[image_id][1]
    width = imgID_2_width_height[image_id][0]

    # 获取FDI编号
    fdi_number = category_id_to_fdi.get(category_id, str(category_id))
    # 解码RLE格式
    rle = {
        "size": size,
        "counts": counts
    }
    binary_mask = maskUtils.decode(rle)

    # 找到边界点
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        for point in contour:
            points.append([int(point[0][0]), int(point[0][1])])

    # 初始化该image_id的结果字典
    if image_id not in image_results:
        image_results[image_id] = {
            "shapes": [],
            "imageHeight": height,
            "imageWidth": width
        }

    # 添加到结果字典的shapes列表
    shape_dict = {
        "label": fdi_number,
        "points": points,
        "score": score
    }
    # image_results[image_id]["shapes"].append(shape_dict)

    # 检查是否已经存在该 label 的 shape
    existing_shapes = image_results[image_id]["shapes"]
    updated = False
    for existing_shape in existing_shapes:
        if existing_shape["label"] == fdi_number:
            # 如果已经存在相同 label 的 shape，则更新分数最高的那个 shape
            if score > existing_shape["score"]:
                existing_shape.update(shape_dict)
            updated = True
            break

    # 如果不存在相同 label 的 shape，则添加新的 shape
    if not updated:
        image_results[image_id]["shapes"].append(shape_dict)
os.makedirs(save_json_path, exist_ok=True)

# # 针对比赛的后处理逻辑
# 确保每个文件名中的每个类别至少有一个box，并保留每个类别中score最高的box
for image_id, result in image_results.items():
    # 为每个类别创建一个字典来存储最终结果
    height = result["imageHeight"]
    width = result["imageWidth"]
    have_class = set()
    final_results = { 
            "shapes": [],
            "imageHeight": height,
            "imageWidth": width
            }

    for shape in result["shapes"]:
        label = shape["label"]
        score = shape["score"]
        points = shape["points"]
        
        # 如果该类别已经有结果，则比较当前box的score是否更高
        if label in have_class:
            current_score = [d for d in final_results['shapes'] if d.get('label', None) == label][0]['score']
            if score > current_score:
                final_results['shapes'] = [d for d in final_results['shapes'] if d.get('label', None) != label]
                final_results['shapes'].append(shape)
        else:
            have_class.add(label)
            final_results['shapes'].append(shape)
    print(len(final_results['shapes']))
    # 将每个image_id的结果写入单独的JSON文件
    file_name = imgID_2_width_height[image_id][2].split('.')[0]
    # output_path = os.path.join(save_json_path, f'{file_name}_Mask.json'.split('STS24_Train_')[1])
    output_path = os.path.join(save_json_path, f'{file_name}_Mask.json')
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)

# 将每个image_id的结果写入单独的JSON文件
# for image_id, result in image_results.items():
#     print(len(result['shapes']))
#     file_name = imgID_2_width_height[image_id][2].split('.')[0]
#     output_path = os.path.join(save_json_path, f'{file_name}_Mask.json'.split('STS24_Train_')[1])
#     with open(output_path, 'w') as f:
#         json.dump(result, f, indent=4)

print(f"All results have been saved to separate JSON files in the '{save_json_path}' directory.")

