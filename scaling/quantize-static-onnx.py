import os
# these two lines are here to hide some warnings from onnxruntime
# TODO: they are not working, I still can see the msgs.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_DISABLE_THREAD_AFFINITY'] = '1'

import yaml

from ultralytics import YOLO

import onnx
from onnxruntime.quantization import quant_pre_process, quantize_static
from onnxruntime.quantization import QuantFormat, QuantType
# from export_torch_to_tflite import convert_onnxfile_to_tflite

from dataset import get_datareader


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

    # Export the model to ONNX
    # best.pt --> best.onnx
    # ref. https://docs.ultralytics.com/modes/export/#arguments
    int8 = False  # Activates INT8 quantization (does not work for ONNX exports)
    simplify = True  # Simplifies the model graph for ONNX exports with onnxslim
    dynamic = False  # Allows dynamic input sizes
    half = False  # Enables FP16 (half-precision) quantization, but only works if device is GPU

    # We export the PyTorch model to an ONNX model
    yolo_model.export(format="onnx", 
                    imgsz=imgsz, 
                    data=config_file,
                    int8=int8, 
                    half=half,  # Enables FP16 (half-precision) quantization
                    simplify=simplify,
                    dynamic=dynamic,
                    )
    # this command raises several pthread_setaffinity_np warnings

    # Load the exported ONNX model
    onnx_model_path = best_model_path.replace(".pt", ".onnx")

    print(f"Loading {onnx_model_path}")
    model = onnx.load(onnx_model_path)

    # # Print the model's inputs - should be just one entry
    # print("Expected inputs:", [input.name for input in model.graph.input])
    input_name = model.graph.input[0].name

    # Print the model's graph structure
    # print(onnx.helper.printable_graph(model.graph))

    datareader = get_datareader(config_file, imgsz=imgsz, limit=10, input_name=input_name)

    # An ONNX preprocessing step, advised before performing quantization
    # best.onnx --> best_pre.onnx
    onnx_model_pre_path = onnx_model_path.replace(".onnx", "_pre.onnx")
    quant_pre_process(
        input_model=onnx_model_path, 
        output_model_path=onnx_model_pre_path,
    )

    # Here we do the actual quantization
    # best_pre.onnx --> best_static.onnx
    onnx_model_quant_static = onnx_model_path.replace("best.onnx", "best_static.onnx")
    quantize_static(
        model_input=onnx_model_pre_path,
        model_output=onnx_model_quant_static,
        calibration_data_reader=datareader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Saved quantized model to {onnx_model_quant_static}")

    # convert_onnxfile_to_tflite(onnx_model_quant_static)

    print("Done")