import argparse
import os
import sys

# access scaling folder
sys.path.append("../scaling")

import yaml
import matplotlib.pyplot as plt

import torch
from ultralytics import YOLO
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import WaterLevelDataset, collate_fn
from pruning_l1_unstructured import count_trainable_parameters, count_active_parameters
import torch_pruning as tp


def parse_opts():
    parser = argparse.ArgumentParser(description="Script configuration")

    parser.add_argument('--weights', type=str, default='../runs/segment/yolo11l-seg-300ep/weights/best.pt', help='relative path to weights file')
    parser.add_argument('--data', type=str, default='../water.yaml', help='relateive path to data.yaml file')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    
    parser.add_argument('--global', dest="global_pruning", action='store_true', help="whether use global or staged (default) pruning")
    # Parse arguments
    args = parser.parse_args()

    return args


def prune_model_global(model, imp, ratio=0.5):
    # I'm not using torch.randn(1, 3, imgsz, imgsz) because it consumes a lot of memory
    example_inputs=torch.randn(1, 3, 32, 32)
    
    # Create pruner
    print("Creating the MagnitudePruner")
    # Make sure the model is in eval mode and gradients are off!
    model.eval()
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        # ch_sparsity=0.5,  # 50% pruning --> This option is deprecated
        pruning_ratio = ratio,  # channel/dim sparsity. you can define the ratio per module with `pruning_ratio_dict`
        # global_pruning = True,  # default is False
        ignored_layers = None,  # list with the nn.Module to ignore
    )
    
    # Apply pruning
    print("Apply the pruning method on the model")
    pruner.step()


def staged_pruning(model, imp, ratio=0.1):
    example_inputs=torch.randn(1, 3, 32, 32)
    print(f"Pruning ratio: {ratio * 100:.2f}%")

    prunable_types = (torch.nn.Conv2d, torch.nn.Linear)
    named_layers = dict(model.named_modules())

    skip_prefixes = ["model.0", "model.1"]  # skip first few blocks

    # Filter only prunable layers with names
    prunable_layers = [(name, module) for name, module in named_layers.items() if isinstance(module, prunable_types) and not any(name.startswith(p) for p in skip_prefixes)]
    
    #  prune each layer individually:
    for name, layer in reversed(prunable_layers):
        print(f"Layer: {name}")
        ignore_layers = [module for n, module in prunable_layers if n != name]
        try:
            # Create a new pruner for this layer only
            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs=example_inputs,
                importance=imp,  # or your custom importance
                pruning_ratio_dict={layer: ratio},  # adjust ratio per layer
                global_pruning=False,
                ignored_layers=ignore_layers,
            )
        
            scores = pruner.importance_scores.get(layer)
            if scores is not None:
                print(f"Layer: {layer.__class__.__name__}, Shape: {scores.shape}")
                print(f"Top 5 importance scores: {scores.topk(5).values.tolist()}")
                print(f"Bottom 5 importance scores: {scores.topk(5, largest=False).values.tolist()}")

            # Apply pruning
            pruner.step()
            print(f"Pruned layer: {layer}")
    
        except Exception as e:
            print(f"Skipping layer due to error: {layer}\n{e}")

    
if __name__ == "__main__": 
    args = parse_opts()
    print(args)
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
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Define importance criterion from torch_pruning
    # p=1: L1 norm
    #p=2: L2 norm
    imp = tp.importance.MagnitudeImportance(p=2)
    # imp = tp.importance.ActivationImportance()
    # imp = tp.importance.GradientImportance()

    if args.global_pruning:
        print("Using global pruning")
        # the code below uses a lot of memory (I tested with 40GB and it is not enough)
        prune_model_global(model, imp)
    else:
        print("Using staged pruning")
        staged_pruning(model, imp)
    
    # We can retrain the pruned model
    # train(model, data_loader, epochs=5)  # TODO: training step

    # check the pruned model
    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")
    