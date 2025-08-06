import os
from tqdm import tqdm
import cv2
import numpy as np
import shutil

def get_files(directory):
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            files += get_files(file_path)
        elif file.endswith('.jpg') or file.endswith('.png'):
            files.append(file_path)
    return files

def seg_to_yolo(mask_path, output_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
    img_height, img_width = mask.shape  # Get image dimensions

    unique_values = np.unique(mask)  # Get unique pixel values representing different classes
    yolo_format_data = []

    for value in unique_values:
        if value == 0:
            continue  # Skip background
        # Create a binary mask for the current class and find contours
        contours, _ = cv2.findContours(
            (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # Find contours

        for contour in contours:
            if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                contour = contour.squeeze()  # Remove single-dimensional entries
                yolo_format = [0] # 0: water class
                for point in contour:
                    # Normalize the coordinates
                    yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                    yolo_format.append(round(point[1] / img_height, 6))
                yolo_format_data.append(yolo_format)
    # Save Ultralytics YOLO format data to file
    with open(output_path, "w", encoding="utf-8") as file:
        for item in yolo_format_data:
            line = " ".join(map(str, item))
            file.write(line + "\n")
    

def seg_to_yolo_fast(mask_path, output_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img_height, img_width = mask.shape

    # Preallocate list
    yolo_lines = []

    # Get contours in one pass for all non-zero values
    non_zero_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(non_zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 3:
            contour = contour.squeeze()
            if contour.ndim != 2:
                continue  # filter out bad contours
            # Normalize with vectorized NumPy (faster)
            norm = contour.astype(np.float32)
            norm[:, 0] /= img_width
            norm[:, 1] /= img_height
            # Flatten and format
            norm_flat = norm.flatten()
            line = '0 ' + ' '.join(f'{v:.6f}' for v in norm_flat)
            yolo_lines.append(line)

    # Write all at once
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines) + '\n')

print("Getting all the files...")
all_files = get_files('.')
images = []
labels = []

for file in tqdm(all_files):
    if 'Annotations' in file:
        labels.append(file)
    elif 'JPEGImages' in file:
        images.append(file)
        
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, 'images'))
    os.makedirs(os.path.join(data_dir, 'labels'))

data = []
print("Copying images to data directory...")
if len(os.listdir(os.path.join(data_dir, 'images'))) == len(images):
    print("Images already copied, skipping...")
else:
    for image in tqdm(images):
        output_path = os.path.join(data_dir, 'images', image[2:].replace('JPEGImages', '-').replace('/', '_'))
        shutil.copy(image, output_path)
        data.append(output_path)
    
print("Converting segment masks to YOLO format...")
for label in tqdm(labels):
    output_path = os.path.join(data_dir, 'labels', label[2:].replace('Annotations', '-').replace('/', '_')[:-3] + 'txt')  # Create output path
    seg_to_yolo(label, output_path)
    
                
f = open('data.txt', 'w')
for file in data:
    f.write(file + '\n')
f.close()