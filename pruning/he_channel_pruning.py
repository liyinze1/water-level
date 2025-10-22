"""

This is a simplified version of the original geometric median pruning algorithm proposed by He et al.


TODO: Test the original implementation on github

---
Ref.

He, Yang, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. 
“Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration.” 
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
IEEE, June 2019, 4335–44. https://doi.org/10.1109/CVPR.2019.00447.

Code: https://github.com/he-y/filter-pruning-geometric-median
""" 
import os
import yaml
import sys
sys.path.append("../scaling")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from ultralytics import YOLO

from pruning_l1_unstructured import count_trainable_parameters, count_active_parameters
from gradcam_pruning import parse_args


def geometric_median_pruning(conv_layer, prune_ratio=0.2, eps=1e-5, max_iter=100):
    W = conv_layer.weight.data  # (out_channels, in_channels, k, k)
    out_channels = W.shape[0]
    W_flat = W.view(out_channels, -1)  # flatten each filter

    # initialize GM as the mean
    gm = W_flat.mean(dim=0, keepdim=True)

    for _ in range(max_iter):
        diff = W_flat - gm
        dist = torch.norm(diff, dim=1, keepdim=True)
        dist[dist < eps] = eps  # avoid division by zero
        weights = 1 / dist
        gm_new = torch.sum(weights * W_flat, dim=0, keepdim=True) / torch.sum(weights)
        if torch.norm(gm_new - gm) < eps:
            break
        gm = gm_new

    # compute distance of each filter to GM
    distances = torch.norm(W_flat - gm, dim=1)
    threshold = torch.quantile(distances, prune_ratio)
    mask = (distances > threshold).float().view(-1, *([1]*(W.dim()-1)))
    conv_layer.weight.data *= mask

    # If layer has bias, mask it too
    if conv_layer.bias is not None:
        conv_layer.bias.data *= mask.view(-1)



if __name__ == "__main__": 
    args = parse_args()
    base_folder = os.path.dirname(__file__)
    best_model_path = os.path.join(base_folder, args.weights)
    config_file = os.path.join(base_folder, args.data)

    transform = v2.Compose([
        v2.Resize([args.imgsz, args.imgsz]),  # we need to resize both h and w
        # v2.ToTensor()  # deprecated, use the line below
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
    ])

    # configuration file
    print(f"Reading {config_file}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # load the best YOLO model
    yolo_model = YOLO(best_model_path)
    model = yolo_model.model  # pytorch model

    total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {total_params}")

    layers = list(model.named_modules())
    print(f"Pruning ratio: {args.ratio}")
    # Traverse in reverse order
    for name, layer in reversed(layers):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            print(f"Pruning layer: {name}")
            geometric_median_pruning(layer, prune_ratio=args.ratio)

    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")

