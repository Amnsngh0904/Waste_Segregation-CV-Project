import os, random, shutil
from pathlib import Path

# Input folder
src = Path("data/waste-collection")

# Output YOLO-style dataset
for split in ["train", "val"]:
    os.makedirs(f"dataset/{split}/images", exist_ok=True)
    os.makedirs(f"dataset/{split}/labels", exist_ok=True)

# Class mapping
class_names = sorted([d.name for d in src.iterdir() if d.is_dir()])
class_to_id = {cls: i for i, cls in enumerate(class_names)}
print("Classes:", class_to_id)

# Train/val split ratio
split_ratio = 0.8

for cls in class_names:
    img_dir = src / cls
    images = list(img_dir.glob("*.*"))  # jpg, png etc
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img in split_imgs:
            # Copy image
            dst_img = Path(f"dataset/{split}/images") / img.name
            shutil.copy(img, dst_img)

            # Create YOLO label (whole image bbox)
            dst_label = Path(f"dataset/{split}/labels") / (img.stem + ".txt")
            with open(dst_label, "w") as f:
                f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")

print("✅ Dataset prepared in YOLO format under dataset/")
