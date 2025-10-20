import os
import sys
# this two lines are here to hide some warnings from onnxruntime
# TODO: they are not working, I still can see the msgs.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_DISABLE_THREAD_AFFINITY'] = '1'

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


def collect_channel_sensitivity(model, data_loader, num_batches=5):
    """
    Collects the sensitivity of each channel in each Conv2d layer in a model.

    Args:
        model (nn.Module): The model to collect sensitivities from.
        data_loader (DataLoader): The data loader to use for collecting sensitivities.
        num_batches (int, optional): The number of batches to use for collecting sensitivities. Defaults to 5.

    Returns:
        dict: A dictionary where the keys are Conv2d layers and the values are the sensitivities of each channel in that layer.
    """
    sensitivity = {}
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(module, torch.nn.Conv2d):
            act = output.detach().abs().mean(dim=(0, 2, 3))  # Mean per channel
            if module not in sensitivity:
                sensitivity[module] = act
            else:
                sensitivity[module] += act

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            model(x)
            if i + 1 == num_batches:
                break

    for h in hooks:
        h.remove()

    # Normalize by number of batches
    for module in sensitivity:
        sensitivity[module] /= num_batches

    return sensitivity


def prune_least_sensitive_channels(sensitivity, prune_ratio=0.2):
    """
    Prune the least sensitive channels in each Conv2d layer in a model.

    Args:
        sensitivity (dict): A dictionary where the keys are Conv2d layers and the values are the sensitivities of each channel in that layer.
        prune_ratio (float, optional): The ratio of channels to prune in each layer. Defaults to 0.2.

    Returns:
        None
    """
    for module, scores in sensitivity.items():
        num_channels = scores.numel()
        num_prune = int(prune_ratio * num_channels)
        if num_prune == 0:
            continue

        # Get indices of least sensitive channels
        prune_idxs = scores.argsort()[:num_prune].tolist()

        # Apply structured pruning
        prune.ln_structured(module, name='weight', amount=len(prune_idxs), n=2, dim=0)



def plot_sensitivity(sensitivity, fname="sentitivity.png"):
    for module, scores in sensitivity.items():
        plt.figure()
        plt.title(f"Sensitivity: {module}")
        plt.plot(scores.numpy())
        plt.xlabel("Channel Index")
        plt.ylabel("Mean Activation")
        plt.savefig(fname)
        plt.close()


if __name__ == "__main__": 
    #
    # definitions
    #
    base_folder = os.path.dirname(__file__)
    imgsz = 640  # the default on train.py
    best_model_path = os.path.join(base_folder, "../runs/segment/yolo11l-seg-300ep/weights/best.pt")
    config_file = os.path.join(base_folder, "../water.yaml")

    transform = v2.Compose([
        v2.Resize([imgsz, imgsz]),  # we need to resize both h and w
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
    sensitivity = collect_channel_sensitivity(model, data_loader, num_batches=5)

    prune_least_sensitive_channels(sensitivity, prune_ratio=0.2)

    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")
    