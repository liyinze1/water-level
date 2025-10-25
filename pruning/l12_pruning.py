"""

This is a simplified version of the original L1/l@ filter pruning algorithm proposed by Li et al., 2017

The main idea is For each convolution layer:

- Each filter = $W_i ∈ \mathbb{R}^{C_{in} × k × k}$
- Compute filter norm:
  - **L1:** $s_i = |W_i|_1 = \\sum |W_i| $
  - **L2:** $s_i = |W_i|_2 = \\sqrt{\\sum W_i^2} $
- Prune filters with smallest norms (lowest contribution).


TODO: Currently we only *zero out* filters; to physically remove them, reconstruct the model using the surviving filters (requires dependency tracking)

---
Ref.
Li, Hao, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf.
“Pruning Filters for Efficient ConvNets.” arXiv preprint arXiv:1608.08710 (2017).
https://arxiv.org/abs/1608.08710

""" 
import os
import yaml
import sys
base_folder = os.path.dirname(__file__)
sys.path.append(os.path.join(base_folder, "../scaling"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from ultralytics import YOLO

from pruning_l1_unstructured import count_trainable_parameters, count_active_parameters
from gradcam_pruning import parse_args



def compute_filter_importance(weight: torch.Tensor, p: int = 1):
    """
    Compute filter importance using Lp norm across (C_in, kH, kW)
    Args:
        weight: (out_channels, in_channels, kH, kW)
        p: 1 for L1 norm, 2 for L2 norm
    Returns:
        importance: (out_channels,) tensor of norms
    """
    return torch.norm(weight.view(weight.size(0), -1), p=p, dim=1)

def prune_conv_layer(conv_layer: nn.Conv2d, prune_ratio=0.2, p=1):
    """
    Prune filters from a single Conv2d layer using L1 or L2 norm.
    Args:
        conv_layer: nn.Conv2d layer to prune
        prune_ratio: fraction of filters to remove (0–1)
        p: 1 for L1 pruning, 2 for L2 pruning
    """
    with torch.no_grad():
        importance = compute_filter_importance(conv_layer.weight.data, p=p)
        threshold = torch.quantile(importance, prune_ratio)
        mask = (importance > threshold).float().view(-1, 1, 1, 1)
        conv_layer.weight.data *= mask  # zero out pruned filters
        if conv_layer.bias is not None:
            conv_layer.bias.data *= mask.view(-1)

def prune_model_l1_l2(model: nn.Module, prune_ratio=0.2, p=1):
    """
    Traverse all Conv2d layers and prune them using L1/L2 filter norm.
    Args:
        model: PyTorch model
        prune_ratio: fraction of filters to prune per layer
        p: 1 (L1) or 2 (L2)
    """
    total = 0
    pruned = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            importance = compute_filter_importance(layer.weight.data, p=p)
            num_prune = int(prune_ratio * importance.numel())
            pruned += num_prune
            total += importance.numel()

            print(f"Pruning {num_prune}/{importance.numel()} filters in {name}")
            prune_conv_layer(layer, prune_ratio, p=p)

    print(f"\nTotal pruned filters: {pruned}/{total} ({100*pruned/total:.2f}%)")



if __name__ == "__main__": 
    args = parse_args()
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

    prune_model_l1_l2(model, args.ratio)

    # Count remaining non-zero trainable parameters
    # remember that this pruning is applied over the weights
    remaining_params = count_active_parameters(model, check_non_zero=True)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    # TODO: check metrics before and after pruning
    # TODO: fine-tunning: retrain for a few epochs to recover accuracy.
    
    print("Done")

