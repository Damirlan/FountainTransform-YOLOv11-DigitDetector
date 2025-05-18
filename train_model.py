from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO("yolo11s.pt")

# Train the model
model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    device="cpu",  # Use CPU instead of GPU
    patience=5,
    name="brake_shoe_numbers"
)

# Save the trained model
model.save("models/brake_shoe_numbers.pt")