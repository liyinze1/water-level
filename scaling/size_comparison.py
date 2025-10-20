import onnx
import numpy as np
import os

# This function loops through all the saved weights and biases tensors in the model by using 'initializer' to calculate the number of parameters in the model
def count_onnx_params(onnx_path):
    """
    Counts the total number of parameters in an ONNX model.
    """
    model = onnx.load(onnx_path)
    total_params = 0
    
    # 'initializer' contains all the stored weights and biases
    for initializer in model.graph.initializer:
        # Get the dimensions of the tensor
        dims = initializer.dims
        
        # Calculate the total number of elements in this tensor
        tensor_params = np.prod(dims)
        
        total_params += tensor_params
        
    return total_params


# We define the model paths
base_folder = os.path.dirname(__file__)
onnx_model_path = os.path.join(base_folder, "../runs/segment/yolo11l-seg-300ep/weights/best.onnx")
quantized_onnx_model_path = os.path.join(base_folder, "../runs/segment/yolo11l-seg-300ep/weights/best_static.onnx")

# We call count_onnx_params to get the number of parameters
fp32_params = count_onnx_params(onnx_model_path)
quant_params = count_onnx_params(quantized_onnx_model_path)

print("--- Model Parameter Count ---")
print(f"FP32 ONNX (best.onnx):     {fp32_params:,} parameters")
print(f"INT8 ONNX (best_static.onnx): {quant_params:,} parameters")

# Now that we have the number of parameters for the fp32 model and the int8 model, we can calculate their size (in bytes)
# We know that for the fp32 model all the parameters are float values (4 bytes)
fp32_size = fp32_params * 4

# For calculating the size of the int8 model, we first need to know the difference in number of parameters between the fp32 model and the int8 model. This difference tels us how many extra parameters are used to store the zero-point and scale values needed for quantization. 
extra_quant_params = quant_params - fp32_params
print(f"EXTRA PARAMS FOR INT8 ONNX: {extra_quant_params:,} parameters")

# Half of the extra parameters will be zero-point and the other half will be scale. Zero-point is an int8 value and scale is an fp32 value. Knowing that all the weights and biases in the quantized model will be int8, we can finally calculate the size of the quantized model as follows:
quant_size = (fp32_params * 1) + ((extra_quant_params/2) * 4) + ((extra_quant_params/2) * 1)

print("--- Model Size ---")
print(f"THE FINAL SIZE OF THE FP32 MODEL:     {fp32_size:,} bytes")
print(f"THE FINAL SIZE OF THE QUANTIZED MODEL: {quant_size:,} bytes")

# We can also calculate the overhead size for each model by comparing the calculated size with the actual file size
fp32_actual_size = os.path.getsize(onnx_model_path)
quant_actual_size = os.path.getsize(quantized_onnx_model_path)

fp32_overhead_size = fp32_actual_size - fp32_size
quant_overhead_size = quant_actual_size - quant_size

print("--- Overhead Size ---")
print(f"THE OVERHEAD SIZE OF THE FP32 MODEL:     {fp32_overhead_size:,} bytes")
print(f"THE OVERHEAD SIZE OF THE QUANTIZED MODEL: {quant_overhead_size:,} bytes")
