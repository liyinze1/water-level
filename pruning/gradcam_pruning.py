import os
import sys
sys.path.append("../scaling")
import argparse
import yaml
import matplotlib.pyplot as plt

import torch
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import WaterLevelDataset, collate_fn
from pruning_l1_unstructured import count_trainable_parameters, count_active_parameters


def parse_args():
    """
    Parse command line arguments and returns an argparse.Namespace object.

    The available arguments are:
        --weights (str): relative path to weights file (default: 'weights/best.pt')
        --data (str): relative path to data.yaml file (default: 'data/water.yaml')
        --imgsz (int): inference size (pixels) (default: 640)
        --ratio (float): how much to prune (between 0 and 1)

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../runs/segment/yolo11l-seg-300ep/weights/best.pt', help='relative path to weights file')
    parser.add_argument('--data', type=str, default='../water.yaml', help='relateive path to data.yaml file')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')

    parser.add_argument('--ratio', type=float, default=0.1, help='pruning ratio')

    args = parser.parse_args()
    return args
    

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

    # data
    dataset = WaterLevelDataset(config_path=config_file, transform=transform)
    # print("Number of images", len(dataset))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Note that batch size is 1, so we can control the number of images that are processed in collect_channel_sensitivity()

    # Identify all Conv2d layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    device = next(model.parameters()).device
    model.eval()

    # Collect channel importance based on Grad-CAM
    channel_scores = {}
    for layer_name, layer in conv_layers:
        cam = GradCAM(model=model, target_layers=[layer])
        scores = torch.zeros(layer.out_channels).to(device)
    
        for images, labels in data_loader:
            images = images.to(device)
            import pdb; pdb.set_trace()
            targets = [ClassifierOutputTarget(label) for label in labels]
    
            _ = cam(input_tensor=images, targets=targets)  # triggers internal activation capture
            activations = cam.activations  # shape: [B, C, H, W]
            grayscale_cam = torch.tensor(cam.last_grayscale_cam).to(device)  # shape: [B, H, W]
    
            for c in range(activations.shape[1]):
                channel_map = activations[:, c, :, :] * grayscale_cam
                scores[c] += channel_map.abs().mean()
    
        scores /= len(data_loader)
        channel_scores[layer_name] = scores.detach().cpu()

    for layer_name, scores in channel_scores.items():
        sorted_indices = torch.argsort(scores)
        num_prune = int(args.ratio * len(sorted_indices))
        print(f"Layer: {layer_name} → Prune channels: {sorted_indices[:num_prune].tolist()}")
    
    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")
    