import os
import numpy as np
from PIL import Image
import yaml

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import Image as TVImage

from onnxruntime.quantization import CalibrationDataReader

class WaterLevelDataset(Dataset):
    """ This class definition is for a Dataset class to provide a dataset for water level segmentation tasks and to train the quantization process.
    """

    def __init__(self, config_path, transform=None):
        """
        Initialize a WaterLevelDataset object.

        Args:
            config_path (str): Path to the YAML configuration file.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
                                            Must be from torchvision.transforms.v2
        """
        self.transform = transform
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        train_fname = os.path.join(os.path.dirname(config_path), config["train"])
        self.data_path = config["path"]
        with open(train_fname, 'r') as f:
            # Read relative paths from train file
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def read_yolo_segmentation_labels(self, label_path):
        """
        Reads a YOLO segmentation label file and returns a list of objects.
        Each object is a tuple: (class_id, polygon), where polygon is a list of (x, y) points.
        Args:
            label_path (str): Path to the YOLO segmentation label file.
        
        Returns:
            list: A list of objects, where each object is a tuple: (class_id, polygon).
        """
        objects = []
        # TXT file with YOLO entries
        # each line contains:
        # <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3 or len(parts) % 2 == 0:
                    continue  # skip malformed lines
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                objects.append((class_id, polygon))
        return objects

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding labels.
            The image is a torch tensor of shape (C, H, W) and the labels are a list of tuples.
            Each tuple contains the class id and the polygon coordinates.
        """
        sample_path = self.samples[idx]

        # RGB image
        img_path = os.path.join(self.data_path, sample_path)
        image = Image.open(img_path)
        image = np.array(image)

        label_path = img_path.replace("data/images", "data/labels")[:-4] + ".txt"
        labels = self.read_yolo_segmentation_labels(label_path)

        if self.transform:
            # prepare image for transformation
            orig_h, orig_w = image.shape[:2]
            image = TVImage(torch.from_numpy(image).permute(2, 0, 1))  # (C, H, W)
            # prepare polygons for transformation
            polygons = []
            class_ids = []
            for class_id, polygon in labels:
                abs_poly = [(x * orig_w, y * orig_h) for x, y in polygon]
                polygons.append(abs_poly)
                class_ids.append(class_id)

            sample = {
                "image": image,
                "annotations": {
                    "polygons": polygons,
                    "labels": torch.tensor(class_ids, dtype=torch.int64)
                }
            }
            sample = self.transform(sample)

            # return the transformed values
            image = torch.as_tensor(sample["image"]).float() / 255.0  # normalize to [0, 1]
            labels = [[label, polygon] for label, polygon in zip(sample["annotations"]["labels"], sample["annotations"]["polygons"])]

        else:
            image = torch.as_tensor(image).float() / 255.0

        return image, labels


# Define a calibration reader
class DataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, limit=None):
        """
        Initializes a DataReader object.

        Args:
            dataloader (DataLoader): The dataloader to read samples from.
            input_name (str): The name of the input to the model.
            limit (int, optional): The maximum number of samples to read. Defaults to None.
        """
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.limit = limit
        self.returned = 0
        self.input_name = input_name
    
    def __len__(self):
        """
        Returns the number of samples in the calibration data reader.

        If a limit is set, the number of samples will be limited to that number.
        Otherwise, the number of samples will be the length of the underlying dataloader.
        """
        if self.limit is not None:
            return min(len(self.dataloader), self.limit)
        else:
            return len(self.dataloader)

    def get_next(self):
        """
        Returns the next sample from the calibration data reader.

        If a limit is set, the data reader will stop returning samples after reaching that limit.
        Otherwise, it will return samples until the end of the underlying dataloader.

        Returns:
            dict or None: A dictionary containing the input data, or None if the end of the data is reached.
        """
        image, label = next(self.iterator)
        self.returned += 1
        if self.limit is not None and self.returned >= self.limit:
            return None  # end of data
        return {self.input_name: image.numpy()}  # image must be a numpy array. don't need labels


def collate_fn(batch):
    """
    Custom collate function to handle variable-length polygon annotations.
    Args:
        batch: List of tuples (image_tensor, label_list)
    Returns:
        images: Tensor of shape (B, C, H, W)
        labels: List of label lists, one per image
    """
    images, labels = zip(*batch)  # unzip list of tuples
    images = torch.stack(images, dim=0)  # stack image tensors
    return images, list(labels)  # keep labels as list of lists


def get_datareader(config_file, imgsz, input_name, limit=10):
    """
    Returns a DataReader object which can be used to provide data to an ONNXRuntime session.

    Args:
        config_file (str): Path to the configuration file (yaml format).
        imgsz (int): Image size.
        input_name (str): Name of the input node in the model.
        limit (int, optional): Maximum number of samples to return. Defaults to 10.

    Returns:
        DataReader: A DataReader object which can be used to provide data to an ONNXRuntime session.
    """
    transform = v2.Compose([
        v2.Resize([imgsz, imgsz]),
        v2.ToTensor()
    ])

    dataset = WaterLevelDataset(config_path=config_file, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Note that batch size is 1

    datareader = DataReader(loader, limit=limit, input_name=input_name)
    return datareader


if __name__ == "__main__":
    # test the behavior of the dataset class
    config_file = os.path.join(os.path.dirname(__file__), "../water.yaml")
    imgsz = 640
    transform = v2.Compose([
        v2.Resize([imgsz, imgsz]),  # we need to resize both h and w
        v2.ToTensor()
    ])

    dataset = WaterLevelDataset(config_path=config_file, transform=transform)
    image, labels = dataset[0]
    print("Image:", image.shape, " objects:", len(labels))

    assert labels[0][0].item() == 0, "Class id is 0"
    assert len(labels[0][1]) == 128, "Polygon with 128 points"
