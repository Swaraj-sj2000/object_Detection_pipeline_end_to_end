import os
root='face_detection_pipeline_end_to_end'

data_path=os.path.join(root,'data')
images_path=os.path.join(data_path,"images")
labels_path=os.path.join(data_path,"labels")


for path in [data_path,images_path,labels_path]:
    if not os.path.exists(path):os.mkdir(path)

aug_data=os.path.join(root,'aug_data')
if not os.path.exists(aug_data):os.mkdir(aug_data)
for part in ['test','train','val']:
    path1=os.path.join(aug_data,part)
    if not os.path.exists(path1):
        os.mkdir(path1)
    for part2 in ['images','labels']:
        path2=os.path.join(path1,part2)
        if not os.path.exists(path2):
            os.mkdir(path2)




    
    
