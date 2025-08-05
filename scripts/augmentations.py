import albumentations as alb
import os
import json
import cv2
import numpy as np

aug_data='aug_data'

augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450, p=1.0), 
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)], 
    bbox_params=alb.BboxParams(format='albumentations', 
    label_fields=['class_labels'] ))

for partition in ['train', 'test', 'val']:
    print(f"{partition} images aug:...")
    for image in os.listdir(os.path.join('data', partition, 'images')):
        print(f"{image} ")
        img = cv2.imread(os.path.join('data', partition, 'images', image))
        
        label_path = os.path.join('data', partition, 'labels', f"{image.split('.')[0]}.json")
        coords = [0] * 4

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            points = np.array(label['shapes'][0]['points'])
            x_min = min(points[0][0], points[1][0]) / 640
            y_min = min(points[0][1], points[1][1]) / 480
            x_max = max(points[0][0], points[1][0]) / 640
            y_max = max(points[0][1], points[1][1]) / 480

            coords = [x_min, y_min, x_max, y_max]


        try:
            for x in range(120):
                print(x)
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

                cv2.imwrite(
                    os.path.join('aug_data', partition, 'images', f"{image.split('.')[0]}.{x}.jpg"),
                    augmented['image']
                )

                annotation = {'image': image}

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0] * 4
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0] * 4
                    annotation['class'] = 0  

                with open(
                    os.path.join('aug_data', partition, 'labels', f"{image.split('.')[0]}.{x}.json"), 'w'
                ) as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)
