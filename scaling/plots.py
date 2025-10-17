import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import torch
import torchvision.transforms.functional as TF


def decode_yolov11_segmentation(outputs, conf_thresh=0.5, image_size=(640, 640)):
    pred = torch.from_numpy(outputs[0][0])  # shape: (37, 8400)
    proto = torch.from_numpy(outputs[1][0])  # shape: (32, 160, 160)

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
    x1 = (cx - w / 2) * image_size[0]
    y1 = (cy - h / 2) * image_size[1]
    x2 = (cx + w / 2) * image_size[0]
    y2 = (cy + h / 2) * image_size[1]
    boxes_pixel = torch.stack([x1, y1, x2, y2], dim=1).int()

    # Reconstruct masks
    masks = torch.einsum("qc,chw->qhw", mask_coeffs.T, proto)  # (N, H, W)
    masks = torch.sigmoid(masks)

    # Resize masks to image size
    masks_resized = TF.resize(masks.unsqueeze(1), image_size, interpolation=TF.InterpolationMode.BILINEAR).squeeze(1)

    return boxes_pixel, masks_resized



def visualize_image_with_mask(image_np, boxes, masks):
    # shows one image and overlays the masks
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    axs = axs.flatten()
    axs[0].imshow(image_np)
    for i in range(len(masks)):
        mask = masks[i]
        axs[1].imshow(mask, alpha=0.4, cmap='Grays')
        x1, y1, x2, y2 = boxes[i]
        axs[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          edgecolor='red', 
                                          facecolor='none', 
                                          linewidth=2))
    for ax in axs:
        ax.axis('off')
    plt.show()
    plt.close(fig)


def overlay_mask_on_image(image, mask, color=(0, 0, 1), alpha=0.6):
    """
    Overlay a binary mask on an image.

    Args:
        image (np.ndarray): Original image in shape (H, W, 3), dtype uint8.
        mask (np.ndarray): Binary mask in shape (H, W), dtype bool or 0/1.
        color (tuple): RGB color for the mask overlay.
        alpha (float): Transparency factor for the overlay.

    Returns:
        np.ndarray: Image with mask overlay.
    """
    overlay = image.copy()
    mask = mask.astype(bool)

    # Create a color layer
    color_layer = np.zeros_like(image)
    color_layer[:,:] = color

    # Blend the original image and the color layer
    overlay[mask] = (1 - alpha) * image[mask] + alpha * color_layer[mask]
    return overlay


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

def overlay_resized_polygons(image, labels, original_size, target_size, color='cyan', linewidth=2):
    """
    Resize and overlay polygons on an image.

    Args:
        image (np.ndarray): Image in (H, W, 3), float in [0, 1].
        labels (List[Tuple[int, List[Tuple[float, float]]]]): List of (class_id, polygon).
        original_size (Tuple[int, int]): Original size of polygons (W, H).
        target_size (Tuple[int, int]): Target image size (W, H).
        color (str or tuple): Color for polygon edges.
        linewidth (int): Thickness of polygon edges.
    """
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    for class_id, polygon in labels:
        if len(polygon) >= 3:
            # Scale polygon points
            scaled_poly = [(x * scale_x, y * scale_y) for x, y in polygon]
            poly_np = np.array(scaled_poly)
            patch = MplPolygon(poly_np, closed=True, edgecolor=color, facecolor='none', linewidth=linewidth)
            ax.add_patch(patch)
            ax.text(poly_np[0, 0], poly_np[0, 1], str(class_id), color=color, fontsize=10, weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.show()
    

def visualize_images_with_masks(images, outputs, labels=None):  
    fig, axs = plt.subplots(len(images), 4, figsize=(4 * 5, 6 + len(images) * 3))   
    for i, [image, [boxes, masks], label] in enumerate(zip(images, outputs, labels)):
        axs[i][0].imshow(image)
        for j in range(len(masks)):
            binary_mask = (masks[j] > 0.5).numpy()

            # prediction
            axs[i][1].imshow(binary_mask, alpha=1.0, cmap='Blues')

            # overlay prediction
            highlighted = overlay_mask_on_image(image, binary_mask, color=(0, 0, 1), alpha=0.5)
            axs[i][2].imshow(highlighted)

        # truth
        axs[i][3].imshow(image)
        color = "cyan"
        linewidth=2
        for class_id, polygon in label:
            if len(polygon) >= 3:  # valid polygon
                poly_np = np.array(polygon)
                patch = MplPolygon(poly_np, closed=True, edgecolor=color, facecolor='none', linewidth=linewidth)
                axs[i][3].add_patch(patch)           

    for i, title in enumerate(["Original", "Predicted", "Overlay", "GT"]):
        axs[0][i].set_title(title)
    
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    