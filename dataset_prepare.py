import os
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

def get_files(directory):
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            files += get_files(file_path)
        elif file.endswith('.jpg') or file.endswith('.png'):
            files.append(file_path)
    return files


all_files = get_files('.')
data = []
annotations = []

for file in all_files:
    if 'Annotations' in file:
        annotations.append(file)
    else:
        data.append(file)
        
print(f"Data files: {len(data)}")
print(f"Annotations files: {len(annotations)}") 

annotations_yolo = []
