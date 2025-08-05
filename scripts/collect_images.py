import cv2
import uuid
import os
import time

cap=cv2.VideoCapture(2) #change device as per required 


num_image=100
data_path="/home/swaraj/sj_code/cv_models/data"
images_path="/home/swaraj/sj_code/cv_models/data/images"
labels_path="/home/swaraj/sj_code/cv_models/data/labels"

#if the file structure somehow is missing itll create it before capturing images and store it in desired folder as mentioned
for path in [data_path,images_path,labels_path]:
    if not os.path.exists(path):
        os.mkdir(path)

#this will add to the images if already present in folder
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
    