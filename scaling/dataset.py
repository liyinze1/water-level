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

    def __init__(self, config_path, transform=None):
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
        return len(self.samples)

    def read_yolo_segmentation_labels(self, label_path):
        """
        Reads a YOLO segmentation label file and returns a list of objects.
        Each object is a tuple: (class_id, polygon), where polygon is a list of (x, y) points.
        """
        objects = []
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
        sample_path = self.samples[idx]

        # RGB image
        img_path = os.path.join(self.data_path, sample_path)
        image = Image.open(img_path)
        image = np.array(image)

        # TXT file with YOLO entries
        # each line contains:
        # <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
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
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.limit = limit
        self.returned = 0
        self.input_name = input_name
    
    def __len__(self):
        if self.limit is not None:
            return min(len(self.dataloader), self.limit)
        else:
            return len(self.dataloader)

    def get_next(self):
        image, label = next(self.iterator)
        self.returned += 1
        if self.limit is not None and self.returned >= self.limit:
            return None  # end of data
        return {self.input_name: image.numpy()}


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
    transform = v2.Compose([
        v2.Resize([imgsz, imgsz]),
        v2.ToTensor()
    ])

    dataset = WaterLevelDataset(config_path=config_file, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

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
