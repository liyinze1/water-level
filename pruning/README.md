# Custom pruning code

- [sensitivity pruning](sensitivity_pruning.py) collects the sensitivity (absolute mean) of each channel in each Conv2d layer in a model and prune the least sensitive channels.

- [he_channel_pruning](he_channel_pruning.py): Uses geometric median pruning applied over the weights

- [pruning_l1_unstructured](pruning_l1_unstructured.py): Use L1 norm to prune network weights

- [lottery_pruning.py](lottery_pruning.py): simplified version of Lottery Ticket Hypothesis pruning proposed by Frankle and Carbin, 2019.

- [taylor_pruning](taylor_pruning.py): Uses simplified version of taylor pruning from Molchanov et al., 2019.

## Not working yet

- [importance_pruning.py](importance_pruning.py): select among several importance metrics to prune the filters

- [l12_pruning.py](l12_pruning.py): simple L1/L2 norm-based filter pruning

- [lth_pruning.py](lth_pruning.py): more complete version of Lottery Ticket Hypothesis pruning

- [gradcam_pruning](gradcam_pruning.py): error processing GradCAM

- [fpgm_pruner](fpgm_pruner.py): raises error while .prune() execution

---

# torch_pruning

This library can be found [here](https://github.com/VainF/Torch-Pruning/). You can see in [tp_example](tp_example.py) how to prune layers using Magnitude Pruning.

**However**, when trying the same pruning using our YOLOv11-based model the model uses too much memory to create the pruner, even if we select a layer at the end of the model and ignore all other Conv2D layers.

> Conclusion: We cannot use this library.