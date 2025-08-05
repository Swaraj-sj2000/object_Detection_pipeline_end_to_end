import cv2
import uuid
import os
import time

cap=cv2.VideoCapture(2)

num_image=100
data_path="/home/swaraj/sj_code/cv_models/data"
images_path="/home/swaraj/sj_code/cv_models/data/images"
labels_path="/home/swaraj/sj_code/cv_models/data/labels"

test_image="/home/swaraj/sj_code/cv_models/data/images/test"
train_image="/home/swaraj/sj_code/cv_models/data/images/train"
val_image="/home/swaraj/sj_code/cv_models/data/images/val"

test_label="/home/swaraj/sj_code/cv_models/data/labels/test"
train_labels="/home/swaraj/sj_code/cv_models/data/labels/train"
val_labels="/home/swaraj/sj_code/cv_models/data/labels/val"


for path in [data_path,images_path,labels_path]:
    if not os.path.exists(path):
        os.mkdir(path)

for i in range(num_image):
    _,frame=cap.read()
    print(f"collecting image {i+1}")
    cv2.imwrite(os.path.join(images_path,f"{str(uuid.uuid1())}.jpg"),frame)
    cv2.imshow('frame',frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    