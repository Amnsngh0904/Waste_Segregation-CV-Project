from ultralytics import YOLO

def run_inference():
    # Load trained model
    model = YOLO("runs/detect/train4/weights/best.pt")

    # Replace with your phone camera IP stream
    phone_cam_url = "http://192.0.0.4:8080/video"  # example

    # Run inference on phone camera stream
    results = model(source=phone_cam_url, show=True, stream=True)

    print("✅ Phone camera inference running... Press 'q' in the window to quit.")

    for r in results:
        pass

if __name__ == "__main__":
    run_inference()
