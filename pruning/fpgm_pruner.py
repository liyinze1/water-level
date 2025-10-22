"""
    Filter Pruning via Geometric Median (FPGM) Structured Pruner

    ref. https://github.com/he-y/filter-pruning-geometric-median
"""

import os
import yaml
import sys
sys.path.append("../scaling")

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.ao.pruning._experimental.pruner import FPGM_pruner
from torchvision.transforms import v2
from ultralytics import YOLO

from dataset import WaterLevelDataset, collate_fn
from pruning_l1_unstructured import count_trainable_parameters, count_active_parameters
from gradcam_pruning import parse_args

if __name__ == "__main__": 
    args = parse_args()
    base_folder = os.path.dirname(__file__)
    best_model_path = os.path.join(base_folder, args.weights)
    config_file = os.path.join(base_folder, args.data)
    device = "cpu"

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

    # data
    dataset = WaterLevelDataset(config_path=config_file, transform=transform)
    # print("Number of images", len(dataset))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Note that batch size is 1, so we can control the number of images that are processed in collect_channel_sensitivity()
    
    # set network-level sparsity: all layers have a sparsity level of 30%
    pruner = FPGM_pruner.FPGMPruner(
        sparsity_level = args.ratio,  # defines the number of filters (rows) that are zeroed-out.
        dist=None,  # default = L2 distance
    )
    
    config = [
        {"tensor_fqn": "model.5.conv.weight"},
        # {"tensor_fqn": "conv2d2.weight"}
    ]
    
    pruner.prepare(model, config)
    pruner.enable_mask_update = True
    pruner.step()
    
    # Get real pruned models (without zeros)
    pruned_model = pruner.prune()  # BUG: raises error

    