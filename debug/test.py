import albumentations as alb
import cv2
import json
import numpy as np
import os

augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450, p=1.0), 
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)], 
    bbox_params=alb.BboxParams(format='albumentations', 
    label_fields=['class_labels'] ))


img_path='images'

print(os.listdir(img_path))

for imgs in os.listdir(img_path):
    print(imgs)
    image=cv2.imread(os.path.join('images',imgs))
    lbl_path=os.path.join('labels',f"{imgs.split('.')[0]}.json")

    with open(lbl_path) as f:
        lbl=json.load(f)

    points = np.array(lbl['shapes'][0]['points'])
    x_min = min(points[0][0], points[1][0]) / 640
    y_min = min(points[0][1], points[1][1]) / 480
    x_max = max(points[0][0], points[1][0]) / 640
    y_max = max(points[0][1], points[1][1]) / 480
    coords = [x_min, y_min, x_max, y_max]
    for i in range(5):
        aug=augmentor(image=image,bboxes=[coords],class_labels=['face'])

        out_img_path=os.path.join('aug_data','images',f"{imgs.split('.')[0]}_{i}.jpg")
        cv2.imwrite(out_img_path,aug['image'])

        annotations={'image':aug['image'].tolist()}
        annotations['bbox']=aug['bboxes'][0]
        annotations['class']=1

 


        with open(os.path.join('aug_data','labels',f"{imgs.split('.')[0]}_{i}.json"),"w") as f:
            json.dump(annotations,f)


                  
    
    


   
    
