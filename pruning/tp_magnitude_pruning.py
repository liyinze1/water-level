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
import torch_pruning as tp



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
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Define importance criterion from torch_pruning
    # p=1: L1 norm
    #p=2: L2 norm
    imp = tp.importance.MagnitudeImportance(p=2)
    # imp = tp.importance.ActivationImportance()
    # imp = tp.importance.GradientImportance()
    
    # Create pruner
    print("Create the MagnitudePruner")
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, imgsz, imgsz),
        importance=imp,
        ch_sparsity=0.5,  # 50% pruning --> This option is deprecated
        # pruning_ratio = 0.5,  # channel/dim sparsity. you can define the ratio per module with `pruning_ratio_dict`
        # global_pruning = True,  # default is False
        # ignored_layers = None,  # list with the nn.Module to ignore
    )
    
    # Apply pruning
    print("Apply the pruning on the model")
    pruner.step()

    # We can retrain the pruned model
    # train(model, data_loader, epochs=5)  # TODO: training step

    # check the pruned model
    remaining_params = count_active_parameters(model)
    print(f"Remaining trainable parameters after pruning: {remaining_params} ({remaining_params / total_params * 100:.2f}%)")

    print("Done")
    