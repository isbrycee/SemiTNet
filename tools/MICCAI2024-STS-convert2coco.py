import os
import json

def convert_to_coco_format(input_folder, output_file):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []  
    }
    annotation_id = 0
    category_id = 1
    label2catid = {}
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(input_folder, json_file), 'r') as f:
            data = json.load(f)
        
        image_id = len(coco_data["images"]) + 1
        
        # 添加图像信息
        image_info = {
            "id": image_id,
            "file_name": data["imagePath"],  
            "height": data["imageHeight"],   
            "width": data["imageWidth"]      
        }
        coco_data["images"].append(image_info)

        # 添加标注信息
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            points_seg = []
            for point in points:
                for item in point:
                    points_seg.append(item)
                
            shape_type = shape["shape_type"]
            
            # 计算外接矩形框的坐标
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            xmin = min(x_coordinates)
            xmax = max(x_coordinates)
            ymin = min(y_coordinates)
            ymax = max(y_coordinates)
            

        
            # 添加类别信息
            if label not in [cat["name"] for cat in coco_data["categories"]]:
                category_info = {
                    "id": category_id,
                    "name": label,
                    "supercategory": "object"
                }
                coco_data["categories"].append(category_info)
                label2catid[label] = category_id
                category_id += 1

            # 构建 COCO 格式中的 annotations 部分
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label2catid[label],
                "segmentation": [points_seg],  
                "area": 0,  
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO 格式的 bbox：[x, y, width, height]
                "iscrowd": 0,
                "ignore": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

# 使用示例：
input_folder = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/datasets/MICCAI2024_SemiTeeth/Train-Labeled/Masks'
output_file = '/root/paddlejob/workspace/env_run/output/haojing/SemiTNet/datasets/MICCAI2024_SemiTeeth/Train-Labeled/train.json'
convert_to_coco_format(input_folder, output_file)
