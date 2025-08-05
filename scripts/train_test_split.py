import os
import random
import shutil

input_path = "data"

train_split = 0.75
test_split = 0.10
val_split = 0.15

#creating three subfolders for data splitting
for part in ['train', 'test', 'val']:
    part_path = os.path.join(input_path, part)
    if not os.path.exists(part_path):
        os.makedirs(os.path.join(part_path, 'images'))
        os.makedirs(os.path.join(part_path, 'labels'))


#reading all images from raw imge directory and shuffeling them for better train,test splitting
image_dir = os.path.join(input_path, 'images')
all_images = os.listdir(image_dir)
all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(all_images)

#deciding what percentage of the shuffled images goes to training, testing and validating
total = len(all_images)
train_count = int(total * train_split)
test_count = int(total * test_split)

train_images = all_images[:train_count]
test_images = all_images[train_count:train_count + test_count]
val_images = all_images[train_count + test_count:]



def copy_files(image_list, subset):
    '''this function reads image names from the list provided and gets corresponding .json file, finally copies both into the corresponsing subfolders as required for splitting'''
    for img_name in image_list:
        label_name = img_name.rsplit('.', 1)[0] + '.json'

        src_img = os.path.join(input_path, 'images', img_name)
        src_lbl = os.path.join(input_path, 'labels', label_name)

        dst_img = os.path.join(input_path, subset, 'images', img_name)
        dst_lbl = os.path.join(input_path, subset, 'labels', label_name)

        shutil.copy(src_img, dst_img)
        if os.path.exists(src_lbl):  
            shutil.copy(src_lbl, dst_lbl)


#copied corresponding images and labels to corresponding subfolders
copy_files(train_images, 'train')
copy_files(test_images, 'test')
copy_files(val_images, 'val')

#after this script the project is ready for augmentations to increase our data set