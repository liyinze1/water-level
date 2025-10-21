#-----------------------------------------------------
#
# convert quantized ONNX model to tflite
#
#-----------------------------------------------------
import os
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def convert_onnxfile_to_tflite(onnx_model_quant_path):
    """
    Convert a quantized ONNX model to a TFLite model.

    Args:
        onnx_model_quant_path: The path to the quantized ONNX model

    Returns:
        None

    Notes:
        This function will save the TFLite model to a file named "model.tflite"
        in the same directory as the ONNX model.
    """
    onnx_model_path = os.path.dirname(onnx_model_quant_path)
    # Load ONNX model
    onnx_model = onnx.load(onnx_model_quant_path)

    # Convert to TensorFlow representation
    tf_rep = prepare(onnx_model)

    # Export as SavedModel
    tf_saved_model_name = os.path.join(onnx_model_path, "tf_saved_model")
    tf_rep.export_graph(tf_saved_model_name)
    print(f"Saved model: {tf_saved_model_name}")

    # use tensorflow to generate TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_name)
    tflite_model = converter.convert()

    # Save to file
    tf_lite_model_name = os.path.join(onnx_model_path, "model.tflite")
    with open(tf_lite_model_name, "wb") as f:
        f.write(tflite_model)
    print(f"Saved tflite model: {tf_lite_model_name}")


def convert_torch_to_tflite(model, imgsz, output_path):
    """
    Convert a PyTorch model to a TFLite model.

    Args:
        model: The PyTorch model to convert
        imgsz: The size of the input image
        output_path: The path where the TFLite model will be saved

    Returns:
        None
    """
    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, imgsz, imgsz)  # adjust shape as needed

    # Export to ONNX
    temp_onnx_file = os.path.join(output_path, "temp_model.onnx")
    torch.onnx.export(model, dummy_input, temp_onnx_file, opset_version=11)

    convert_onnxfile_to_tflite(temp_onnx_file)
    # os.remove(temp_onnx_file)  # Optional: delete the temporary ONNX file
    
