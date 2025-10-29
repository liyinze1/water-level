"""
    Example of MagnitudePruner on a resnet18 model

""" 

import torch
from torchvision.models import resnet18
import torch_pruning as tp


if __name__ == "__main__":
    model = resnet18(pretrained=True)

    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.TaylorImportance()

    # Ignore some layers, e.g., the output layer
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    # Initialize a pruner
    iterative_steps = 5 # progressive pruning
    print("MagnitudePruner creation...")
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    # prune the model, iteratively if necessary.
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        print("Step", i)
        # Taylor expansion requires gradients for importance estimation
        if isinstance(imp, tp.importance.TaylorImportance):
            # A dummy loss, please replace it with your loss function and data!
            loss = model(example_inputs).sum() 
            loss.backward() # before pruner.step()

        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        # finetune your model here
        # finetune(model)
        # ...
        