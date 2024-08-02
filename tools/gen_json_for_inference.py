import os
import json
import cv2

def generate_coco_template(train_json_path, image_folder, output_file):
    # 读取原始的 train.json 文件作为模板
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    # 提取 categories 字段
    categories = train_data.get("categories", [])
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 1
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            
            # 使用 OpenCV 读取图片尺寸
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            
            # 构建每张图片的信息
            image_info = {
                "id": image_id,
                "file_name": filename,
                "height": height,
                "width": width
            }
            coco_data["images"].append(image_info)
            
            image_id += 1
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

# 使用示例：
train_json_path = 'Train-Labeled/train.json'
image_folder = 'Validation-Public'
output_file = 'validation.json'

generate_coco_template(train_json_path, image_folder, output_file)
