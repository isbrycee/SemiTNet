import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import torch
from torchvision.ops import nms

def apply_nms(detections, iou_threshold=0.5):
    # Convert detections to tensors
    box_list = []
    for det in detections:
        box_xywh = det['bbox']
        xmin, ymin = box_xywh[0], box_xywh[1]
        xmax, ymax = xmin + box_xywh[2], ymin + box_xywh[3]
        box_list.append((xmin, ymin, xmax, ymax))

    boxes = torch.tensor(box_list, dtype=torch.float)

    scores = torch.tensor([det['score'] for det in detections])
    indices = nms(boxes, scores, iou_threshold)

    # Filter detections
    filtered_detections = [detections[i] for i in indices]
    return filtered_detections

def compute_metrics(gt_json, results_json, iou_threshold=0.5):
    # Load ground truth and results
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(results_json)

    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.array([iou_threshold])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract bbox precision, recall, f1
    precision = coco_eval.stats[1]
    recall = coco_eval.stats[8]
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f'BBox Precision: {precision:.4f}')
    print(f'BBox Recall: {recall:.4f}')
    print(f'BBox F1 Score: {f1:.4f}')

    # Mask-level metrics
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.iouThrs = np.array([iou_threshold])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mask-level iou and dice similarity
    precision = coco_eval.stats[1]
    recall = coco_eval.stats[8]
    iou = coco_eval.stats[0]
    dice = 2 * precision * recall / (precision + recall)

    print(f'Mask IoU: {iou:.4f}')
    print(f'Mask Dice Similarity: {dice:.4f}')

def compute_metrics_for_each_class(gt_json, results_json, iou_threshold=0.5, score_threshold=0.0, nms_iou_threshold=0.5):
    # Load ground truth
    coco_gt = COCO(gt_json)
    
    # Load and filter results
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    filtered_results = [res for res in results if res['score'] >= score_threshold]

    # Apply NMS for each image
    img_ids = coco_gt.getImgIds()
    filtered_results_by_img = defaultdict(list)
    for res in filtered_results:
        filtered_results_by_img[res['image_id']].append(res)
    
    final_results = []
    for img_id, detections in filtered_results_by_img.items():
        final_results.extend(apply_nms(detections, nms_iou_threshold))

    # Save filtered results to a temporary file
    filtered_results_json = 'filtered_results.json'
    with open(filtered_results_json, 'w') as f:
        json.dump(final_results, f)
    
    coco_dt = coco_gt.loadRes(filtered_results_json)

    # Divide images into two groups based on the number of bounding boxes
    img_ann_counts = {img_id: len(coco_gt.getAnnIds(imgIds=img_id)) for img_id in img_ids}
    group1 = [img_id for img_id in img_ids if img_ann_counts[img_id] < 32]
    group2 = [img_id for img_id in img_ids if img_ann_counts[img_id] == 32]

    # Function to calculate metrics for each group
    def calculate_metrics(group, group_name):
        # Create new COCO objects for the group
        coco_gt_group = COCO()
        coco_dt_group = COCO()

        # Filter annotations and images
        coco_gt_group.dataset = {
            'images': [img for img in coco_gt.dataset['images'] if img['id'] in group],
            'annotations': [ann for ann in coco_gt.dataset['annotations'] if ann['image_id'] in group],
            'categories': coco_gt.dataset['categories']
        }
        coco_dt_group.dataset = {
            'images': [img for img in coco_dt.dataset['images'] if img['id'] in group],
            'annotations': [ann for ann in coco_dt.dataset['annotations'] if ann['image_id'] in group],
            'categories': coco_dt.dataset['categories']
        }
        coco_gt_group.createIndex()
        coco_dt_group.createIndex()

        # Initialize COCOeval object
        coco_eval = COCOeval(coco_gt_group, coco_dt_group, 'bbox')
        coco_eval.params.iouThrs = np.array([iou_threshold])
        coco_eval.params.imgIds = group
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract category-wise metrics
        gt_counts = defaultdict(int)
        correct_counts = defaultdict(int)

        for cat in coco_gt_group.dataset['categories']:
            cat_id = cat['id']
            gt_counts[cat_id] = len(coco_gt_group.getAnnIds(catIds=[cat_id]))
        for eval_img in coco_eval.evalImgs:
            if eval_img is None:
                continue
            matched_gts = set()
            for i, dt_match in enumerate(eval_img['dtMatches'][0]):
                if dt_match and dt_match not in matched_gts:
                    cat_id = eval_img['category_id']
                    correct_counts[cat_id] += 1
                    matched_gts.add(dt_match)

        print(f'\nMetrics for {group_name}:')
        for cat in coco_gt_group.dataset['categories']:
            cat_id = cat['id']
            cat_name = cat['name']
            print(f'Category {cat_name} (ID: {cat_id}):')
            print(f'  GT Count: {gt_counts[cat_id]}')
            wrong_percentage = round((gt_counts[cat_id] - correct_counts[cat_id]/4)/gt_counts[cat_id], 3) * 100
            print(f'  Wrong Count: {wrong_percentage}%')

    # Calculate metrics for each group
    calculate_metrics(group1, 'Group 1 (boxes < 32)')
    calculate_metrics(group2, 'Group 2 (boxes = 32)')

def compute_metrics_for_fully_partial(gt_json, results_json, iou_threshold=0.5):
    # Load ground truth and results
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(results_json)

    # Get image ids
    img_ids = coco_gt.getImgIds()

    # Initialize variables to store results
    metrics_part1_bbox = {'precision': [], 'recall': [], 'f1': []}
    metrics_part1_mask = {'precision': [], 'recall': [], 'iou': [], 'dice': []}
    metrics_part2_bbox = {'precision': [], 'recall': [], 'f1': []}
    metrics_part2_mask = {'precision': [], 'recall': [], 'iou': [], 'dice': []}
    metrics_part3_bbox = {'precision': [], 'recall': [], 'f1': []}
    metrics_part3_mask = {'precision': [], 'recall': [], 'iou': [], 'dice': []}

    # Loop over each image
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        annotations = coco_gt.loadAnns(ann_ids)
        dt_ann_ids = coco_dt.getAnnIds(imgIds=img_id)
        dt_annotations = coco_dt.loadAnns(dt_ann_ids)


        num_gt_boxes = len(ann_ids)

        if num_gt_boxes < 32:
            # Calculate metrics for part 1
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.iouThrs = np.array([iou_threshold])
            coco_eval.params.imgIds = [img_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[0]  # precision at IoU threshold
            recall = coco_eval.stats[8]     # recall at IoU threshold
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics_part1_bbox['precision'].append(precision)
            metrics_part1_bbox['recall'].append(recall)
            metrics_part1_bbox['f1'].append(f1)

            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.params.iouThrs = np.array([iou_threshold])
            coco_eval.params.imgIds = [img_id]
            # coco_eval.params.useCats = False
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[0]  # precision at IoU threshold
            recall = coco_eval.stats[8]     # recall at IoU threshold
            iou = coco_eval.stats[0]
            dice = 2 * precision * recall / (precision + recall)
            metrics_part1_mask['precision'].append(precision)
            metrics_part1_mask['recall'].append(recall)
            metrics_part1_mask['iou'].append(iou)
            metrics_part1_mask['dice'].append(dice)

        elif num_gt_boxes == 32:
            # Calculate metrics for part 2
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.iouThrs = np.array([iou_threshold])
            coco_eval.params.imgIds = [img_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[0]  # precision at IoU threshold
            recall = coco_eval.stats[8]     # recall at IoU threshold
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics_part2_bbox['precision'].append(precision)
            metrics_part2_bbox['recall'].append(recall)
            metrics_part2_bbox['f1'].append(f1)

            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.params.iouThrs = np.array([iou_threshold])
            coco_eval.params.imgIds = [img_id]
            # coco_eval.params.useCats = False
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[0]  # precision at IoU threshold
            recall = coco_eval.stats[8]     # recall at IoU threshold
            iou = coco_eval.stats[0]
            dice = 2 * precision * recall / (precision + recall)
            metrics_part2_mask['precision'].append(precision)
            metrics_part2_mask['recall'].append(recall)
            metrics_part2_mask['iou'].append(iou)
            metrics_part2_mask['dice'].append(dice)

        elif num_gt_boxes > 32:
            # Calculate metrics for part 1
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.iouThrs = np.array([iou_threshold])
            coco_eval.params.imgIds = [img_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[0]  # precision at IoU threshold
            recall = coco_eval.stats[8]     # recall at IoU threshold
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics_part3_bbox['precision'].append(precision)
            metrics_part3_bbox['recall'].append(recall)
            metrics_part3_bbox['f1'].append(f1)

            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.params.iouThrs = np.array([iou_threshold])
            coco_eval.params.imgIds = [img_id]
            # coco_eval.params.useCats = False
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[0]  # precision at IoU threshold
            recall = coco_eval.stats[8]     # recall at IoU threshold
            iou = coco_eval.stats[0]
            dice = 2 * precision * recall / (precision + recall)
            metrics_part3_mask['precision'].append(precision)
            metrics_part3_mask['recall'].append(recall)
            metrics_part3_mask['iou'].append(iou)
            metrics_part3_mask['dice'].append(dice)

    # Calculate average metrics for part 1
    avg_precision_part1_bbox = np.mean(metrics_part1_bbox['precision'])
    avg_recall_part1_bbox = np.mean(metrics_part1_bbox['recall'])
    avg_f1_part1_bbox = 2 * (avg_precision_part1_bbox * avg_recall_part1_bbox) / (avg_precision_part1_bbox + avg_recall_part1_bbox)

    avg_precision_part1_mask = np.mean(metrics_part1_mask['precision'])
    avg_recall_part1_mask = np.mean(metrics_part1_mask['recall'])
    avg_iou_part1_mask = np.mean(metrics_part1_mask['iou'])
    avg_dice_part1_mask = np.mean(metrics_part1_mask['dice'])

    # Calculate average metrics for part 2
    avg_precision_part2_bbox = np.mean(metrics_part2_bbox['precision'])
    avg_recall_part2_bbox = np.mean(metrics_part2_bbox['recall'])
    avg_f1_part2_bbox = 2 * (avg_precision_part2_bbox * avg_recall_part2_bbox) / (avg_precision_part2_bbox + avg_recall_part2_bbox)

    avg_precision_part2_mask = np.mean(metrics_part2_mask['precision'])
    avg_recall_part2_mask = np.mean(metrics_part2_mask['recall'])
    avg_iou_part2_mask = np.mean(metrics_part2_mask['iou'])
    avg_dice_part2_mask = np.mean(metrics_part2_mask['dice'])

    # Calculate average metrics for part 3
    avg_precision_part3_bbox = np.mean(metrics_part3_bbox['precision'])
    avg_recall_part3_bbox = np.mean(metrics_part3_bbox['recall'])
    avg_f1_part3_bbox = 2 * (avg_precision_part3_bbox * avg_recall_part3_bbox) / (avg_precision_part3_bbox + avg_recall_part3_bbox)

    avg_precision_part3_mask = np.mean(metrics_part3_mask['precision'])
    avg_recall_part3_mask = np.mean(metrics_part3_mask['recall'])
    avg_iou_part3_mask = np.mean(metrics_part3_mask['iou'])
    avg_dice_part3_mask = np.mean(metrics_part3_mask['dice'])

    # Print or return the results
    print("Metrics for images with < 32 ground truth boxes:")
    print(f"BBox Precision: {avg_precision_part1_bbox:.4f}")
    print(f"BBox Recall: {avg_recall_part1_bbox:.4f}")
    print(f"BBox F1 Score: {avg_f1_part1_bbox:.4f}")
    print(f"Mask IoU: {avg_iou_part1_mask:.4f}")
    print(f"Mask Dice Similarity: {avg_dice_part1_mask:.4f}")
    print()

    print("Metrics for images with 32 ground truth boxes:")
    print(f"BBox Precision: {avg_precision_part2_bbox:.4f}")
    print(f"BBox Recall: {avg_recall_part2_bbox:.4f}")
    print(f"BBox F1 Score: {avg_f1_part2_bbox:.4f}")
    print(f"Mask IoU: {avg_iou_part2_mask:.4f}")
    print(f"Mask Dice Similarity: {avg_dice_part2_mask:.4f}")
    print()

    print("Metrics for images with > 32 ground truth boxes:")
    print(f"BBox Precision: {avg_precision_part3_bbox:.4f}")
    print(f"BBox Recall: {avg_recall_part3_bbox:.4f}")
    print(f"BBox F1 Score: {avg_f1_part3_bbox:.4f}")
    print(f"Mask IoU: {avg_iou_part3_mask:.4f}")
    print(f"Mask Dice Similarity: {avg_dice_part3_mask:.4f}")
    print()

if __name__ == '__main__':
    gt_json = 'tooth-x-ray-instance-segmentation-1.6k/tooth_IS_test.json'
    results_json = 'coco_instances_results.json'

    # print("Compute Metrics:")
    # compute_metrics(gt_json, results_json)
    # print()
    
    print("Compute Metrics For Partial Images:")
    compute_metrics_for_fully_partial(gt_json, results_json)
    print()

    # print("Compute Metrics For Each Class:")
    # compute_metrics_for_each_class(gt_json, results_json)
    # print()
