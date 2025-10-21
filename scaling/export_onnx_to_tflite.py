import os
import argparse
from ultralytics import YOLO


def parse_args():
    """
    Parse command line arguments and returns an argparse.Namespace object.

    The available arguments are:
        --weights (str): relative path to weights file (default: 'weights/best.pt')
        --data (str): relative path to data.yaml file (default: 'data/water.yaml')
        --imgsz (int): inference size (pixels) (default: 640)
        --int8 (bool): if True, quantize to int8 (default: True)

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='relative path to weights file')
    parser.add_argument('--data', type=str, default='data/water.yaml', help='relateive path to data.yaml file')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')

    parser.add_argument('--int8', action='store_true', help='quantize to int8')
    args = parser.parse_args()
    return args


def export_onnx_to_tflite(model, config_file, int8: bool = True, imgsz: int = 640):
    """
    Export the model to TFLite INT8 format.

    Args:
        model (YOLO): The model to export.
        config_file (str): Path to your dataset YAML.
        int8 (bool, optional): If True, quantize to int8. Defaults to True.
        imgsz (int, optional): Inference size (pixels). Defaults to 640.

    Returns:
        None
    """
    # Export to TFLite INT8
    # ref. https://docs.ultralytics.com/modes/export/#usage-examples
    model.export(
        format='tflite',
        int8=int8,
        data= config_file, # Path to your dataset YAML
        imgsz=imgsz
    )
    # the file is saved in the same folder as the original model with a .tflite extension
    print("Export to TFLite INT8 complete. Check the weights folder.")


if __name__ == "__main__":
    args = parse_args()

    base_folder = os.path.dirname(__file__)
    model_path = os.path.join(base_folder, args.weights)
    config_file = os.path.join(base_folder, args.data)

    # Load your trained model
    model = YOLO(model_path)

    export_onnx_to_tflite(model, config_file, int8=args.int8, imgsz=args.imgsz)
