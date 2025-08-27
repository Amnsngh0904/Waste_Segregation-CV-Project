# from ultralytics import YOLO

# def train_model():
#     # Load a small YOLOv8 model (lightweight, good for MacBook Air M4)
#     model = YOLO("yolov8n.pt")  

#     # Train on your dataset
#     model.train(
#         data="data.yaml",   # dataset config
#         epochs=25,          # increase if needed
#         imgsz=224,          # smaller image size = faster training
#         batch=16,
#         device="mps" 
#     )

# if __name__ == "__main__":
#     train_model()

from ultralytics import YOLO
import albumentations as A

def train_model():
    # Load YOLOv8 small model
    model = YOLO("yolov8n.pt")

    # Define Albumentations augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(p=0.4),
        A.MotionBlur(p=0.2),
        A.RandomFog(p=0.2),
        A.RandomRain(p=0.2),
        A.Cutout(num_holes=3, max_h_size=20, max_w_size=20, p=0.5)
    ])

    # Train on your dataset with Albumentations
    model.train(
    data="data.yaml",
    epochs=25,
    imgsz=224,
    batch=16,
    device="mps",
    augment=True,    # enables YOLO augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
)

if __name__ == "__main__":
    train_model()
