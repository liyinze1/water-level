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

And unzip the customised dataset `roadrunner_photos`

### 4. Train
```bash
python train.py
```
