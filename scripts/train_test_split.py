import os
import random
import shutil

DATA_DIR = "data"
RAW_IMG_DIR = os.path.join(DATA_DIR, "images")
RAW_LABEL_DIR = os.path.join(DATA_DIR, "labels")

SPLITS = {
    "train": 0.75,
    "test": 0.10,
    "val": 0.15
}

def ensure_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(DATA_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, "labels"), exist_ok=True)

def get_image_list():
    images = [f for f in os.listdir(RAW_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(images)
    return images

def split_data(images):
    total = len(images)
    train_count = int(SPLITS["train"] * total)
    test_count = int(SPLITS["test"] * total)

    return {
        "train": images[:train_count],
        "test": images[train_count:train_count + test_count],
        "val": images[train_count + test_count:]
    }

def copy_split_files(image_list, split_name):
    for img_name in image_list:
        label_name = os.path.splitext(img_name)[0] + ".json"

        src_img = os.path.join(RAW_IMG_DIR, img_name)
        src_lbl = os.path.join(RAW_LABEL_DIR, label_name)

        dst_img = os.path.join(DATA_DIR, split_name, "images", img_name)
        dst_lbl = os.path.join(DATA_DIR, split_name, "labels", label_name)

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

if __name__ == "__main__":
    print("[INFO] Creating directories...")
    ensure_dirs()

    print("[INFO] Collecting and splitting image data...")
    all_images = get_image_list()
    splits = split_data(all_images)

    for split_name, image_list in splits.items():
        print(f"[INFO] Copying {len(image_list)} files to '{split_name}' folder...")
        copy_split_files(image_list, split_name)

    print("[INFO] Dataset split complete. You can now run augmentations.")
