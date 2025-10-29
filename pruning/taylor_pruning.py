"""
This is an adaptation of the method proposed by Molchanov et al.
Theirs is an interactive pruning that consists of the following steps:
1) Fine-tune the network until convergence on the target task; 
2) Alternate iterations of pruning and further fine-tuning; 
3) Stop pruning after reaching the target trade-off between accuracy and pruning
objective, e.g. floating point operations (FLOPs) or memory utilization.

TODO: implement the stopping condition in (3)


Compute the importance of each filter based on the **first-order Taylor expansion** of the loss w.r.t. that filter.
For filter ( i ):

$\\Delta L_i \\approx \\left| \\frac{\\partial L}{\\partial a_i} \\cdot a_i \\right|$

Where ( a_i ) is the output feature map of filter ( i ).

---
Ref.

Molchanov, Pavlo, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. 2017. 
“Pruning Convolutional Neural Networks for Resource Efficient Inference.” 
arXiv preprint arXiv:1611.06440. https://arxiv.org/abs/1611.06440

"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../scaling"))

import yaml

import torch
from torch import nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from ultralytics import YOLO

from dataset import WaterLevelDataset, collate_fn
from pruning_l1_unstructured import count_trainable_parameters, count_active_parameters
from gradcam_pruning import parse_args
from evaluate import create_binary_mask, decode_yolov11_segmentation2, combine_masks


def taylor_importance(conv_layer):
    # conv_layer.weight.grad shape: (out_channels, in_channels, k, k)
    grad = conv_layer.weight.grad
    if grad is None:
        return None
    weight = conv_layer.weight
    importance = torch.sum(torch.abs(grad * weight), dim=(1,2,3))  # sum over in_channels, kernel
    return importance



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    base_folder = os.path.dirname(__file__)
    best_model_path = os.path.join(base_folder, args.weights)
    config_file = os.path.join(base_folder, args.data)

    transform = v2.Compose([
        v2.Resize([args.imgsz, args.imgsz]),  # we need to resize both h and w
        # v2.ToTensor()  # deprecated, use the line below
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True)  # notice that v2.ToImage converts to [0.0, 1.0]
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

    # Note that this code assumes that **batch size is 1**, 
    # so we can control the number of images that are processed below
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  

    # loss function
    criterion = nn.CrossEntropyLoss()
    # from the DataLoader
    inputs, targets = next(iter(data_loader))

    model.train()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True    
    outputs = model(inputs.to(device))

    # need to reshape outputs, because with model.train() the output is different from model.eval()
    img_shape = targets[0][0][1].canvas_size  # (h, w), all images have the same shape
    # decode output into a mask
    boxes, masks = decode_yolov11_segmentation2(outputs, device=device)
    if len(masks) == 0:
        # TODO: raise error
        print("No masks found")
        exit(0)
    
    # decode input into a mask
    pred_mask = combine_masks(masks)
    target_mask = create_binary_mask(targets, img_shape)

    print("Getting gradients")
    print("pred_mask shape:", pred_mask.shape, pred_mask.requires_grad)
    loss = criterion(pred_mask, target_mask.to(device))
    print(f"Loss value: {loss.item()}")
    model.zero_grad()
    loss.backward()
    # for name, param in model.named_parameters():
    #     grad_status = 'has grad' if param.grad is not None else 'no grad'
    #     print(f"{name}: {grad_status}")

    # compute Taylor importance per filter
    layers = list(model.named_modules())
    # Traverse in reverse order
    for name, layer in reversed(layers):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):

            importance = taylor_importance(layer)
            if importance is None:
                # print(f"⚠ Pruning layer: {name} --> No gradients for this layer")
                continue
            else:
                print(f"Pruning layer: {name}")
            threshold = torch.quantile(importance, args.ratio)  # prune ratio% least important
            mask = (importance > threshold).float().view(-1,1,1,1)
            layer.weight.data *= mask  # prune weights

            # TODO: check stop condition
    
    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")
