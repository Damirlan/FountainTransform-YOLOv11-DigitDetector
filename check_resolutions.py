import cv2
from pathlib import Path

image_dir = Path("data")
resolutions = set()
for img_path in image_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        resolutions.add((w, h))
print("Unique resolutions:", resolutions)