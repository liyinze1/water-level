import os
from tqdm import tqdm
import cv2
import numpy as np

def get_files(directory):
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            if file == 'Annotations':
                new_name = os.path.join(directory, 'labels')
                os.rename(file_path, new_name)
                file_path = new_name
            elif file == 'JPEGImages':
                new_name = os.path.join(directory, 'images')
                os.rename(file_path, new_name)
                file_path = new_name
            files += get_files(file_path)
        elif file.endswith('.jpg') or file.endswith('.png'):
            files.append(file_path)
    return files

def seg_to_yolo(mask_path):
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
    output_path = mask_path[:-3] + 'txt'  # Change extension to txt
    with open(output_path, "w", encoding="utf-8") as file:
        for item in yolo_format_data:
            line = " ".join(map(str, item))
            file.write(line + "\n")


print("Getting all the files...")
all_files = get_files('.')
images = []
labels = []

for file in tqdm(all_files):
    if 'labels' in file:
        labels.append(file)
    elif 'images' in file:
        images.append(file)
        
print("Converting segment masks to YOLO format...")
for label in tqdm(labels):
    seg_to_yolo(file)
                
print(f"Data files: {len(labels)}")
print(f"Annotations files: {len(images)}") 

f = open('data.txt', 'w')
for file in images:
    f.write(file + '\n')
f.close()