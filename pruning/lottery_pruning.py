"""
Lottery Ticket Hypothesis (Frankle & Carbin, 2019) proposes that:

A randomly-initialized dense neural network contains a subnetwork (a "winning ticket") that, when trained in isolation, can match the performance of the full model.

Ref:
The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.” 
arXiv preprint arXiv:1803.03635 (2019).
https://arxiv.org/abs/1803.03635

"""
import os
import argparse
import yaml
import sys
base_folder = os.path.dirname(__file__)
sys.path.append(os.path.join(base_folder, "../scaling"))

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from ultralytics import YOLO

from pruning_l1_unstructured import count_trainable_parameters  # , count_active_parameters
from dataset import WaterLevelDataset, collate_fn
from evaluate import create_binary_mask, decode_yolov11_segmentation2, combine_masks


def move_to_device(obj, device):
    """
    Recursively move a torch.Tensor or a list of tensors to a given device.

    If the input is a torch.Tensor, it is moved to the given device using the
    `to` method.

    If the input is a list, this function is called recursively on each element of the
    list.

    If the input is of any other type, it is returned unchanged.

    Args:
        obj: The object to move to the device.
        device: The device to move the object to.

    Returns:
        The moved object.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    else:
        return obj  # pass through other types unchanged


# -------------------------
# Training and evaluation
# -------------------------

def train(model, loader, optimizer, criterion, epoch, device):
    """
    Train a model for one epoch.

    Args:
        model: The model to train.
        loader: The DataLoader containing the training data.
        optimizer: The optimizer to use for training.
        criterion: The loss function to use for training.
        epoch: The current epoch number.
        device: The device to use for training (e.g. a CUDA device).

    Returns:
        None
    """
    model.train()
    model.to(device)
    total_loss = 0
    for inputs, targets in tqdm(loader, desc=f"Training epoch {epoch}"):
        inputs = inputs.to(device)  # [B, C, H, W]
        targets = move_to_device(targets, device)
        img_shape = inputs.shape[2:]  # (h, w), all images have the same shape

        optimizer.zero_grad()
        outputs = model(inputs)

        # decode output into a mask
        boxes, masks = decode_yolov11_segmentation2(outputs, device=device)
        if len(masks) == 0:
            # TODO: raise error
            print("No masks found")
            exit(0)
        
        # decode input into a mask
        pred_mask = combine_masks(masks, img_shape)
        target_masks = create_binary_mask(targets, img_shape)

        loss = criterion(pred_mask, target_masks.to(device))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Loss: {total_loss / len(loader)}")


def test(model, loader, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Data loader with the dataset to evaluate on.
        device (torch.device): Device to evaluate on.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.eval()
    correct = 0
    with torch.no_grad():

        for inputs, targets in tqdm(loader, desc="Test"):
            inputs = inputs.to(device)  # [B, C, H, W]
            targets = move_to_device(targets, device)
            img_shape = inputs.shape[2:]  # (h, w), all images have the same shape

            outputs = model(inputs)

            # decode output into a mask
            boxes, masks = decode_yolov11_segmentation2(outputs, device=device)
            if len(masks) == 0:
                # TODO: raise error
                print("No masks found")
                exit(0)
            
            # decode input into a mask
            pred_mask = combine_masks(masks, img_shape).to(device)
            target_masks = create_binary_mask(targets, img_shape).to(device)
            correct += (pred_mask == target_masks).sum().item() / pred_mask.numel()

    return correct / len(loader.dataset)

# -------------------------
# Pruning Function
# -------------------------
def prune_by_magnitude(model, prune_percent):
    # Flatten all weights
    all_weights = torch.cat([param.view(-1).abs() for name, param in model.named_parameters() if "weight" in name])
    threshold = torch.quantile(all_weights, prune_percent)

    mask_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = (param.abs() > threshold).float()
            mask_dict[name] = mask
    return mask_dict

# -------------------------
# Apply Mask
# -------------------------
def apply_mask(model, mask_dict):
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data *= mask_dict[name]



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
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('-b', '--batch-size', type=int, default=10, help='Batch size for training')

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    # -------------------------
    # Main LTH workflow
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    best_model_path = os.path.join(base_folder, args.weights)
    config_file = os.path.join(base_folder, args.data)

    transform = v2.Compose([
        v2.Resize([args.imgsz, args.imgsz]),  # we need to resize both h and w
        # v2.ToTensor()  # deprecated, use the line below
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
    ])

    # data
    train_dataset = WaterLevelDataset(config_path=config_file, transform=transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)  

    test_dataset = WaterLevelDataset(config_path=config_file, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  

    # configuration file
    print(f"Reading {config_file}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # load the best YOLO model
    yolo_model = YOLO(best_model_path)
    model = yolo_model.model  # pytorch model

    total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {total_params}")


    initial_state = {k: v.clone() for k, v in model.state_dict().items()}  # save initial weights

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, device)
    acc_before = test(model, test_loader, device)  # TODO: add other metrics
    print(f"Accuracy before pruning: {acc_before:.4f}")

    # Prune 'ratio' % of weights
    mask_dict = prune_by_magnitude(model, prune_percent=args.ratio)
    apply_mask(model, mask_dict)

    # Reset weights to original initialization
    model.load_state_dict(initial_state)
    apply_mask(model, mask_dict)  # reapply pruning mask

    # Retrain the sparse subnetwork
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, device)
    acc_after = test(model, test_loader, device)
    print(f"Accuracy after retraining pruned model: {acc_after:.4f}")
