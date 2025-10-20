import os
import sys
# this two lines are here to hide some warnings from onnxruntime
# TODO: they are not working, I still can see the msgs.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_DISABLE_THREAD_AFFINITY'] = '1'

sys.path.append("../scaling")

import yaml

import torch
from ultralytics import YOLO
import torch.nn.utils.prune as prune


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.

    This function iterates over the model's parameters and sets them all to be trainable.
    It then returns the total number of trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to count the trainable parameters.

    Returns
    -------
    int
        The total number of trainable parameters in the model.
    """
    # make the model trainable so we can count the number of trainable parameters
    for param in model.parameters():
        param.requires_grad = True    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_active_parameters(model):
    remaining_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Check if pruning was applied
            if hasattr(module, 'weight_mask'):
                # Count only unmasked (non-zero) weights
                remaining_params += torch.count_nonzero(module.weight_mask)
            else:
                # If not pruned, count all weights
                remaining_params += module.weight.numel()
    return remaining_params



if __name__ == "__main__": 
    #
    # definitions
    #
    base_folder = os.path.dirname(__file__)
    imgsz = 640  # the default on train.py
    best_model_path = os.path.join(base_folder, "../runs/segment/yolo11l-seg-300ep/weights/best.pt")
    config_file = os.path.join(base_folder, "../water.yaml")

    # configuration file
    print(f"Reading {config_file}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # load the best YOLO model
    yolo_model = YOLO(best_model_path)

    dummy_input = torch.rand(size=(1, 3, imgsz, imgsz))
    model = yolo_model.model
    
    total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {total_params}")

    model.eval()
    # L1 pruning
    # check https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html

    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            print("L1 Pruning", name)
            prune.l1_unstructured(module, name='weight', amount=0.2)
        # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            print("L1 Pruning", name)
            prune.l1_unstructured(module, name='weight', amount=0.4)

    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")
