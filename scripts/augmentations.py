import albumentations as A
import os
import json
import cv2
import numpy as np

# Directory where augmented data will be saved
AUG_DATA_DIR = 'aug_data'
ORIG_DATA_DIR = 'data'

# Define augmentation pipeline
augmentor = A.Compose([
    A.RandomCrop(width=450, height=450, p=1.0), 
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.RGBShift(p=0.2),
    A.VerticalFlip(p=0.5)
], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

def normalize_bbox(points, img_width=640, img_height=480):
    x_min = min(points[0][0], points[1][0]) / img_width
    y_min = min(points[0][1], points[1][1]) / img_height
    x_max = max(points[0][0], points[1][0]) / img_width
    y_max = max(points[0][1], points[1][1]) / img_height
    return [x_min, y_min, x_max, y_max]

def augment_partition(partition):
    print(f"\n[INFO] Augmenting '{partition}' partition...")
    image_dir = os.path.join(ORIG_DATA_DIR, partition, 'images')
    label_dir = os.path.join(ORIG_DATA_DIR, partition, 'labels')
    save_img_dir = os.path.join(AUG_DATA_DIR, partition, 'images')
    save_lbl_dir = os.path.join(AUG_DATA_DIR, partition, 'labels')

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    for image_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, f"{os.path.splitext(image_file)[0]}.json")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {image_file}")
            continue

        coords = [0, 0, 0, 0]
        has_label = os.path.exists(label_path)

        if has_label:
            with open(label_path, 'r') as f:
                label = json.load(f)
            points = np.array(label['shapes'][0]['points'])
            coords = normalize_bbox(points)

        for i in range(120):
            try:
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

                # Save image
                aug_image_name = f"{os.path.splitext(image_file)[0]}.{i}.jpg"
                cv2.imwrite(os.path.join(save_img_dir, aug_image_name), augmented['image'])

                # Save annotation
                annotation = {
                    'image': aug_image_name,
                    'bbox': [0, 0, 0, 0],
                    'class': 0
                }

                if has_label and augmented['bboxes']:
                    annotation['bbox'] = augmented['bboxes'][0]
                    annotation['class'] = 1

                with open(os.path.join(save_lbl_dir, f"{os.path.splitext(image_file)[0]}.{i}.json"), 'w') as f:
                    json.dump(annotation, f)

            except Exception as e:
                print(f"[ERROR] Failed on {image_file} ({i}): {e}")

if __name__ == "__main__":
    for part in ['train', 'test', 'val']:
        augment_partition(part)
