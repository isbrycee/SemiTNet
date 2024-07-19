import json
import os
import cv2
import random
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
import matplotlib.colors as mplc
from tqdm import tqdm

COCO_CATEGORIES = [
    {"color": [128, 64, 128], "isthing": 1, "id": 1, "name": "1"},
   {"color": [44, 35, 232], "isthing": 1, "id": 2, "name": "2"},
   {"color": [70, 70, 70], "isthing": 1, "id": 3, "name": "4"},
   {"color": [102, 102, 156], "isthing": 1, "id": 4, "name": "5"},
   {"color": [190, 153, 153], "isthing": 1, "id": 5, "name": "6"},
   {"color": [153, 153, 153], "isthing": 1, "id": 6, "name": "7"},
   {"color": [250, 170, 30], "isthing": 1, "id": 7, "name": "8"},
   {"color": [220, 220, 0], "isthing": 1, "id": 8, "name": "9"},
   {"color": [107, 142, 35], "isthing": 1, "id": 9, "name": "10"},
   {"color": [152, 251, 152], "isthing": 1, "id": 10, "name": "11"},
   {"color": [70, 130, 180], "isthing": 1, "id": 11, "name": "12"},
   {"color": [220, 20, 60], "isthing": 1, "id": 12, "name": "13"},
   {"color": [255, 0, 0], "isthing": 1, "id": 13, "name": "15"},
   {"color": [0, 0, 142], "isthing": 1, "id": 14, "name": "16"},
   {"color": [0, 0, 70], "isthing": 1, "id": 15, "name": "17"},
   {"color": [0, 60, 100], "isthing": 1, "id": 16, "name": "19"},
   {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "20"},
   {"color": [0, 0, 230], "isthing": 1, "id": 18, "name": "21"},
   {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "22"},
   {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "23"},
   {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "24"},
   {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "25"},
   {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "26"},
   {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "27"},
   {"color": [119, 11, 32], "isthing": 1, "id": 25, "name": "28"},
   {"color": [72, 0, 118], "isthing": 1, "id": 26, "name": "30"},
   {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "32"},
   {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "18"},
   {"color": [209, 0, 151], "isthing": 1, "id": 29, "name": "29"},
   {"color": [188, 208, 182], "isthing": 1, "id": 30, "name": "3"},
   {"color": [0, 220, 176], "isthing": 1, "id": 31, "name": "14"},
   {"color": [255, 99, 164], "isthing": 1, "id": 32, "name": "31"},
]


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 32, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    return thing_colors

# Function to load results
def load_coco_results(results_json):
    with open(results_json) as f:
        results = json.load(f)
    return results

# Function to load ground truth dataset
def load_gt_dataset(gt_json):
    with open(gt_json) as f:
        gt_data = json.load(f)
    return gt_data

# Function to register the dataset
def register_dataset(name, json_file, image_root):
    register_coco_instances(name, {}, json_file, image_root)

# Function to convert the results to detectron2 format
def convert_to_detectron2_format(results, gt_data, categories):
    dataset_dicts = []
    image_id_to_filename = {img['id']: img['file_name'] for img in gt_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    image_annotations = {}
    for result in results:
        image_id = result['image_id']
        if image_id not in image_id_to_filename:
            continue
        if result['score'] < 0.5:
            continue
        if image_id not in image_annotations:
            image_annotations[image_id] = {
                'file_name': image_id_to_filename[image_id],
                'image_id': image_id,
                'annotations': []
            }
        
        obj = {
            'bbox': result['bbox'],
            'bbox_mode': BoxMode.XYWH_ABS,
            'segmentation': result.get('segmentation', []),
            'category_id': result['category_id']
        }
        image_annotations[image_id]['annotations'].append(obj)
    
    for image_id, data in image_annotations.items():
        dataset_dicts.append(data)
    
    return dataset_dicts

# Function to convert the ground truth to detectron2 format
def convert_gt_to_detectron2_format(gt_data):
    dataset_dicts = []
    for img in gt_data['images']:
        record = {}
        record["file_name"] = img["file_name"]
        record["image_id"] = img["id"]
        record["height"] = img["height"]
        record["width"] = img["width"]
        
        annos = []
        for anno in gt_data['annotations']:
            if anno['image_id'] == img['id']:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": anno["category_id"],
                    "segmentation": anno.get("segmentation", [])
                }
                annos.append(obj)
        
        record["annotations"] = annos
        dataset_dicts.append(record)
    
    return dataset_dicts

def _jitter(color):
    """
    Randomly modifies given color to produce a slightly different color than the color given.

    Args:
        color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
            picked. The values in the list are in the [0.0, 1.0] range.

    Returns:
        jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
            color after being jittered. The values in the list are in the [0.0, 1.0] range.
    """
    color = mplc.to_rgb(color)
    vec = np.random.rand(3)
    # better to do it in another color space
    vec = vec / np.linalg.norm(vec) * 0.5
    res = np.clip(vec + color, 0, 1)
    return tuple(res)

# Function to visualize and save the results
# def visualize_and_save_results(dataset_dicts, metadata, output_dir, image_root):
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate random colors for each category
#     category_colors = {int(category): [random.randint(0, 255) for _ in range(3)] for category in metadata.thing_classes}
#     colors = [
#                     _jitter([int(x) / 255 for x in category_colors[c]])
#                     for c in category_colors.keys()
#                 ]
#     for idx, d in enumerate(dataset_dicts):
#         img = cv2.imread(os.path.join(image_root, d["file_name"]))
#         visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
#         out = visualizer.draw_dataset_dict(d, colors)

#         output_path = os.path.join(output_dir, f"visualization_{idx}.png")
#         cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

def visualize_and_save_results(dataset_dicts, metadata, output_dir, image_root):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random colors for each category

    for idx, d in tqdm(enumerate(dataset_dicts)):
        img = cv2.imread(os.path.join(image_root, d["file_name"]))
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        out = visualizer.draw_dataset_dict(d)

        output_path = os.path.join(output_dir, f"visualization_{idx}.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    # Define paths
    results_json = 'coco_instances_results.json'
    gt_json = 'datasets/tooth-x-ray-instance-segmentation-1.6k/tooth_IS_test.json'
    image_root = 'datasets/tooth-x-ray-instance-segmentation-1.6k/img/'
    output_dir = 'visual_res/visual_semitnet/'
    # output_dir_gt = 'visual_gt'

    # Register the dataset
    dataset_name = 'my_dataset'
    register_dataset(dataset_name, gt_json, image_root)

    # Load metadata
    metadata = MetadataCatalog.get(dataset_name)

    # Load ground truth data
    gt_data = load_gt_dataset(gt_json)

    # Extract categories from ground truth data
    categories = gt_data.get('categories', [])
    category_names = [cat['name'] for cat in categories]

    # Add thing_classes to metadata
    MetadataCatalog.get(dataset_name).thing_classes = category_names
    MetadataCatalog.get(dataset_name).thing_colors = _get_coco_instances_meta()

    # Load results
    results = load_coco_results(results_json)

    # Convert results to detectron2 format
    dataset_dicts_results = convert_to_detectron2_format(results, gt_data, categories)
    
    # Convert ground truth to detectron2 format
    dataset_dicts_gt = convert_gt_to_detectron2_format(gt_data)

    # Visualize and save results
    visualize_and_save_results(dataset_dicts_results, metadata, output_dir, image_root)
    
    # Visualize and save ground truth
    # visualize_and_save_results(dataset_dicts_gt, metadata, output_dir_gt, image_root)

