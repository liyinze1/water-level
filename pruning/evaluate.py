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
from ultralytics.utils.tal import make_anchors, dist2bbox, TaskAlignedAssigner



def bbox_decode(anchor_points: torch.Tensor, pred_dist: torch.Tensor, proj, use_dfl: bool = True) -> torch.Tensor:
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    if use_dfl:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
    return dist2bbox(pred_dist, anchor_points, xywh=False)


def decode_yolov11_segmentation(outputs, conf_thresh=0.5, image_size=(640, 640)):
    """
    Decodes the output of the YOLO model into bounding boxes and segmentation masks.

    Parameters:
    outputs (list): The output of the YOLO model, which contains 2 entries.
    conf_thresh (float, optional): The minimum confidence score for a bounding box to be considered valid. Defaults to 0.5.
    image_size (tuple, optional): The size of the input image. Defaults to (640, 640).

    Returns:
    tuple: A tuple containing the bounding boxes and segmentation masks. The bounding boxes are represented as a tensor of shape (N, 4) where N is the number of bounding boxes. The segmentation masks are represented as a tensor of shape (N, H, W) where H and W are the height and width of the input image, respectively.
    """
    # YOLO outputs variable is different if model is .train() or .eval()
    feats, pred_masks, proto = outputs if len(outputs) == 3 else outputs[1]
    batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width

    pred = outputs[0][0].detach().cpu()  # shape: (37, 8400)
    proto = outputs[1][2][0].detach().cpu()  # shape: (32, 160, 160)

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


def decode_yolov11_segmentation2(
    outputs,  # output from yolov11 model's model (i.e. outputs = yolov11.model(inputs))
    conf_thresh=0.5, 
    device = "cpu",
    nc = 1,  # yolo_model.model.nc
    reg_max = 16,  # see https://docs.ultralytics.com/reference/nn/modules/head/#ultralytics.nn.modules.head.WorldDetect.forward
    stride = [8, 16, 32],  # model.stride
    ):
    """
    Decodes the output of the YOLO model into bounding boxes and segmentation masks.

    Parameters:
    outputs (list): The output of the YOLO model, which contains 2 or 3 entries.
    conf_thresh (float, optional): The minimum confidence score for a bounding box to be considered valid. Defaults to 0.5.
    image_size (tuple, optional): The size of the input image. Defaults to (640, 640).

    Returns:
    tuple: A tuple containing the bounding boxes and segmentation masks. The bounding boxes are represented as a tensor of shape (N, 4) where N is the number of bounding boxes. The segmentation masks are represented as a tensor of shape (N, H, W) where H and W are the height and width of the input image, respectively.
    """
    # YOLO outputs variable is different if model is .train() or .eval()
    feats, pred_masks, proto = outputs if len(outputs) == 3 else outputs[1]
    batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width

    no = nc + reg_max * 4    
    pred_distri, pred_scores = torch.cat(
        [xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split((reg_max * 4, nc), 1
    )

    # B, grids, ..
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    pred_masks = pred_masks.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    image_size = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]  # image size (h,w)
    image_size =  [int(x) for x in list(image_size)]

    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
    proj = torch.arange(reg_max, dtype=torch.float, device=device)

    pred_bboxes = bbox_decode(anchor_points, pred_distri, proj, True)  # xyxy, (b, h*w, 4)

    # Filter by confidence
    keep = (pred_scores > 0.5).squeeze(-1).squeeze(0)  # select only objects with score > 0.5 and return shape (8400,)
    boxes = pred_bboxes[:, keep].squeeze(0)  # shape: (n, 4)

    # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = (cx - w / 2)
    y1 = (cy - h / 2)
    x2 = (cx + w / 2)
    y2 = (cy + h / 2)
    boxes_pixel = torch.stack([x1, y1, x2, y2], dim=1).int()

    # Reconstruct masks
    mask_coeffs = pred_masks[:, keep].squeeze(0).detach().cpu()  # shape: (1, n, 32)
    pred_mask = torch.einsum('ni,jihw->nhw', mask_coeffs, proto)
    masks = torch.sigmoid(pred_mask)  # (N, H, W)

    # Resize masks to image size
    # NOTE: masks_resized must be a tensor, and keep its gradient
    masks_resized = TF.resize(masks.unsqueeze(1), image_size, interpolation=TF.InterpolationMode.BILINEAR).squeeze(1)

    # masks as binary
    masks_resized[masks_resized > .5] = 1
    masks_resized[masks_resized <= .5] = 0

    return boxes_pixel, masks_resized


def combine_masks(masks: torch.Tensor) -> torch.Tensor:
    combined_mask = masks[0].float()
    for i in range(1, masks.shape[0]):
        combined_mask = combined_mask + masks[i].float()
    
    # need to clamp it to keep it binary
    combined_mask = combined_mask.clamp(0, 1)
    return combined_mask


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
    if isinstance(imgsz, int): 
        # expand imgsz
        imgsz = (imgsz, imgsz)  

    # Initialize empty mask
    mask = np.zeros(imgsz, dtype=np.uint8)

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

