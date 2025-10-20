import json
import os

def labelme_to_yolo(json_file, output_dir):
    """
    Convert LabelMe JSON format to YOLO format (single class)
    
    Args:
        json_file: Path to LabelMe JSON file
        output_dir: Directory to save YOLO format txt files
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename (same as image name but with .txt extension)
    base_name = os.path.splitext(data['imagePath'])[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")
    
    # Convert annotations
    yolo_lines = []
    for shape in data['shapes']:
        points = shape['points']
        
        # Single class, always use class ID 0
        class_id = 0
        
        # Normalize coordinates
        normalized_points = []
        for x, y in points:
            norm_x = x / img_width
            norm_y = y / img_height
            normalized_points.extend([norm_x, norm_y])
        
        # Create YOLO format line: class_id x1 y1 x2 y2 x3 y3 ...
        yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
        yolo_lines.append(yolo_line)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    print(f"Converted {len(yolo_lines)} annotations")
    print(f"Output saved to: {output_file}")


def main():
    source_folder = "roadrunner_photos/json"
    output_folder = "roadrunner_photos/labels"
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(source_folder, filename)
            labelme_to_yolo(json_path, output_folder)
    
    images_folder = "roadrunner_photos/images"
    f = open('data.txt', 'a')
    for filename in os.listdir(images_folder):
        f.write(f"{os.path.join(images_folder, filename)}\n")
    f.close()
    
if __name__ == "__main__":
    main()