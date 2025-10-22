# Custom pruning code

- [sensitivity pruning](sensitivity_pruning.py) collects the sensitivity (absolute mean) of each channel in each Conv2d layer in a model and prune the least sensitive channels.


- [gradcam_pruning](gradcam_pruning.py): error processing GradCAM

- [he_channel_pruning](he_channel_pruning.py): running but it seems like it is not pruning any layer

- [taylor_pruning](taylor_pruning.py): model is not returning the gradients. thus, the method cannot be executed. returns same results as he_channel_pruning.

- [fpgm_pruner](fpgm_pruner.py): 

# torch_pruning

This library can be found [here](https://github.com/VainF/Torch-Pruning/). You can see in [tp_example](tp_example.py) how to prune layers using Magnitude Pruning.

**However**, when trying the same pruning using our YOLOv11-based model the model uses too much memory to create the pruner, even if we select a layer at the end of the model and ignore all other Conv2D layers.

> Conclusion: We cannot use this library.