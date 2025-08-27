from ultralytics import YOLO

def evaluate_model():
    # Load trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Run evaluation on validation dataset
    metrics = model.val()

    print("✅ Evaluation complete.")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
