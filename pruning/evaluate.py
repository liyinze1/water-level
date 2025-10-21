"""
Consider you create a model that receives an images (B, 3, imgsz, imgsz) using the following commands:

yolo_model = YOLO(best_model_path)
model = yolo_model.model

# if you evaluate one image
outputs = model(dummy_input)
boxes, masks = decode_yolov11_segmentation(results, image_size=list(tensor.shape[2:]))

---

the output is composed of several parts as shown below:

outputs: 2 entries
outputs[0].shape: torch.Size([1, 37, 8400])
outputs[1]: 3 entries
outputs[1][0]     : 3 entries
- outputs[1][0][0]: torch.Size([1, 65, 80, 80])
- outputs[1][0][1]: torch.Size([1, 65, 40, 40])
- outputs[1][0][2]: torch.Size([1, 65, 20, 20])
outputs[1][1]     : 1 entries
- outputs[1][1][0]: torch.Size([32, 8400])
outputs[1][2]       : 1 entries
  - outputs[1][2][0]: torch.Size([32, 160, 160])
  
"""
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF


def decode_yolov11_segmentation(outputs, conf_thresh=0.5, image_size=(640, 640)):
    pred = outputs[0][0]  # shape: (37, 8400)
    proto = outputs[1][2][0] # shape: (32, 160, 160)

    # Extract components
    boxes = pred[0:4, :]  # x, y, w, h
    obj_scores = pred[4, :]  # objectness
    mask_coeffs = pred[5:, :]  # (32, 8400)

    # Filter by confidence
    keep = obj_scores > conf_thresh
    boxes = boxes[:, keep]
    mask_coeffs = mask_coeffs[:, keep]

    # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = (cx - w / 2)
    y1 = (cy - h / 2)
    x2 = (cx + w / 2)
    y2 = (cy + h / 2)
    boxes_pixel = torch.stack([x1, y1, x2, y2], dim=1).int()

    # Reconstruct masks
    masks = torch.einsum("qc,chw->qhw", mask_coeffs.T, proto)  # (N, H, W)
    masks = torch.sigmoid(masks)

    # Resize masks to image size
    masks_resized = TF.resize(masks.unsqueeze(1), image_size, interpolation=TF.InterpolationMode.BILINEAR).squeeze(1)

    # masks as binary
    masks_resized[masks_resized > .5] = 1
    masks_resized[masks_resized <= .5] = 0

    return boxes_pixel, masks_resized


def combine_masks(masks):
    mask = np.zeros_like(masks[0]).astype(bool)
    for i in range(masks.shape[0]):
        block_mask = masks[0].numpy()
        block_mask = block_mask.astype(bool)
        # Create a color layer
        mask = np.logical_or.reduce([mask, block_mask])
    return mask


def decode_raw_labels(lines):
    # lines contains the lines from the label.txt file. Each line consists
    # class_id x1 y1 x2 y2 x3 y3 ....
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3 or len(parts) % 2 == 0:
            continue  # skip malformed lines
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        labels.append((class_id, polygon))
    return labels


def resize_polygons(labels, target_size):
    """
    Resize polygons to image size.

    Args:
        labels (List[Tuple[int, List[Tuple[float, float]]]]): List of (class_id, polygon).
        the polygon coordinates are normalized (i.e., between 0 and 1).
        target_size (Tuple[int, int]): Target image size (W, H) to scale the polygons to the image size
    """
    scale_x = target_size[0]
    scale_y = target_size[1]

    scaled_labels = []
    for class_id, polygon in labels:
        if len(polygon) >= 3:
            # Scale polygon points
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            scaled_labels.append([class_id, scaled_poly])
            
    return scaled_labels


def create_binary_mask(scaled_labels, imgsz, selected_class_id=0):
    # Initialize empty mask
    mask = np.zeros((imgsz, imgsz), dtype=np.uint8)

    for class_id, polygon in scaled_labels:
        if class_id != selected_class_id:
            continue

        # Convert polygon to integer coordinates
        pts = np.array(polygon, dtype=np.int32)

        # Reshape for fillPoly: expects shape (num_polygons, num_points, 2)
        pts = pts.reshape((-1, 1, 2))

        # Fill polygon on the mask
        cv2.fillPoly(mask, [pts], color=1)

    return mask  # dtype=uint8, values 0 or 1


def compute_binary_metrics(pred_mask, label_mask):
    pred = torch.tensor(pred_mask, dtype=torch.bool).detach().clone()
    target = torch.tensor(label_mask, dtype=torch.bool).detach().clone()

    TP = torch.logical_and(pred == 1, target == 1).sum().item()
    TN = torch.logical_and(pred == 0, target == 0).sum().item()
    FP = torch.logical_and(pred == 1, target == 0).sum().item()
    FN = torch.logical_and(pred == 0, target == 1).sum().item()

    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    sensitivity = recall
    f1_score    = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    dice        = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    iou         = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    results = {
        "True Positives (TP)": TP,
        "True Negatives (TN)": TN,
        "False Positives (FP)": FP,
        "False Negatives (FN)": FN,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "F1 Score": f1_score,
        "Dice Coefficient": dice,
        "Intersection over Union (IoU)": iou
    }

    return results

