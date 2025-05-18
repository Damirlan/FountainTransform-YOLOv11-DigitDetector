import cv2
import numpy as np
from pathlib import Path
import os

# Paths
input_dir = Path("data")
output_dir = Path("data_sl_processed")
images_output_dir = output_dir / "images"
labels_output_dir = output_dir / "labels"

# Step 1: Create output directories
images_output_dir.mkdir(parents=True, exist_ok=True)
labels_output_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Function to resize with padding
def resize_with_padding(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    top = (target_size[1] - new_h) // 2
    left = (target_size[0] - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = resized
    return padded

# Step 3: Function to enhance image quality
def enhance_image(image):
    # Convert to grayscale for enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram equalization for contrast
    equalized = cv2.equalizeHist(gray)
    img_enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    # Sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_enhanced, -1, kernel)
    return img_sharpened

# Step 4: Process images
image_paths = list(input_dir.glob("*.jpg"))
resolutions = set()
low_resolution_images = []
processed_count = 0

for img_path in image_paths:
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue

    # Log resolution
    h, w = img.shape[:2]
    resolutions.add((w, h))
    if w * h < 100000:  # Arbitrary threshold for "low resolution" (e.g., < 100k pixels)
        low_resolution_images.append(img_path.name)

    # Resize with padding
    img_resized = resize_with_padding(img)

    # Enhance image
    img_enhanced = enhance_image(img_resized)

    # Save processed image
    output_path = images_output_dir / img_path.name
    cv2.imwrite(str(output_path), img_enhanced)
    processed_count += 1

# Step 5: Summary
print("\nDataset Preprocessing Summary:")
print(f"Total images processed: {processed_count}")
print(f"Unique resolutions before preprocessing: {len(resolutions)}")
print(f"Resolution range: {min(resolutions, key=lambda x: x[0]*x[1])} to {max(resolutions, key=lambda x: x[0]*x[1])}")
print(f"Potential low-resolution images to review: {len(low_resolution_images)}")
if low_resolution_images:
    print("Low-resolution images:", low_resolution_images[:5], "..." if len(low_resolution_images) > 5 else "")
print(f"Processed dataset saved to: {output_dir}")
print(f"Images directory: {images_output_dir}")
print(f"Labels directory (empty, ready for annotations): {labels_output_dir}")
print("\nNext steps: Annotate images using LabelImg, then split into train/val sets.")