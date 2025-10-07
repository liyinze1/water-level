### 1. Download the dataset

```bash
curl -L -o ./water-segmentation-dataset.zip https://www.kaggle.com/api/v1/datasets/download/gvclsu/water-segmentation-dataset

unzip water-segmentation-dataset.zip

mv water_v2/water_v2/* water_v2
mv water_v1/water_v1/* water_v1
```

### 2. Install dependency

```bash
pip install ultralytics
pip install wandb
```
https://github.com/ultralytics/ultralytics

### 3. Prepare the dataset to YOLO format
```bash
python dataset_prepare.py
```
This script will read all the image files (png and jpg), pairing the image with its label, convert the label to YOLO format, store the images with labels under `./data` and put the pathes of all images into `./data.txt`

```bash
python split.py
```
This script will split the dataset by 8:2, and output `water_train.txt`, `water_val.txt` and `water.yaml`

### 4. Train
```bash
python train.py
```

### 5. About river only images
```bash
python split_river.py
```
In the dataset, "ADE20K" folder contains any water related images. This script will create `river_train.txt`, `river_val.txt` and `river.yaml` by excluding "ADE20k"
