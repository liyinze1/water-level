#!/usr/bin/env python3
"""
Simple photo enhancement to boost saturation
"""

from PIL import Image, ImageEnhance
import sys
import os

def enhance_saturation(image_path, saturation_boost=1.5):
    """
    Enhance saturation of all colors in an image
    """
    # Open image
    img = Image.open(image_path)
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(img)
    enhanced_img = enhancer.enhance(saturation_boost)
    
    # Save enhanced image
    output_path = image_path.replace('.png', '_enhanced.png').replace('.jpg', '_enhanced.jpg').replace('.jpeg', '_enhanced.jpeg')
    enhanced_img.save(output_path, quality=95)
    print(f"Enhanced image saved: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    image_path = sys.argv[1]
    saturation_boost = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    
    image_path_list = []
    
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        for file in os.listdir(image_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path_list.append(os.path.join(image_path, file))
    else:
        image_path_list.append(image_path)
    
    try:
        for image_path in image_path_list:
            enhance_saturation(image_path, saturation_boost)
    except Exception as e:
        print(f"Error: {e}")