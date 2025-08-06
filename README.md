# FACE DETECTION PIPELINE
This is a beginner-friendly, end-to-end face detection pipeline built using TensorFlow, OpenCV, and Python-only tools.
It covers data collection, data augmentation, preprocessing, model building, and inference, all organized in a modular, reusable package structure

# Features
-Capture and label your own dataset using webcam

-Apply rich augmentations (brightness, contrast, rotation, etc.)

-Preprocess the dataset for training

-Build and train a custom face detection model

-Make predictions on new images

-All functionality packed into a reusable Python package


## Project Structure
FACE_DETECTION_PIPELINE_END_TO_END/

    aug_data/ ### Prepared data after augmentations
        test/
            images/
            labels/
        train/
            images/
            labels/
        val/
            images/
            labels/

    data/ ### the raw image data before augmentation
        images/
        labels/
        test/
        train/
        val/

    documents/ ### contains the explanation diagrams and other documents like logs and training performace etc.
        logs/
        requirements.txt
        project_explanation_diagram.png

    face_detection/ ### the core package
        __init__.py
        config.py
        dataset_loader.py
        losses.py
        model.py
        utils.py

    notebooks/
        main.ipynb ### example notebook for testing

    

    scripts/ ### scripts files to run at different stages and programmes
        augmentaions.py
        collect_images.py
        make_projects.py
        train_test_split.py
        train.py

    README.md
    setup.py #

## Installations:
1.clone the repo

git clone https://github.com/Swaraj-sj2000/object_Detection_pipeline_end_to_end.git

cd FACE_DETECTION_PIPELINE_END_TO_END

pip install -e .

pip install notebook

2.Getting Started

jupyter notebook

### Step 1: Collect and label images using webcam
python scripts/collect_images.py

### Step 2: Apply augmentations
python scripts/augmentaions.py

### Step 3: Split dataset into train/val/test
python scripts/train_test_split.py

### Step 4: Train the model
python scripts/train.py


# Model Overview
The model used here is a basic CNN built with TensorFlow and Keras. 
It's designed to detect any object  from images using a bounding box regression approach but here i have used it to detect my own face.
You can modify model.py to improve performance using custom architectures or pre-trained 'VGG16' backbones.

# Modules Explanations:
-face_detection/config.py: Configuration and hyperparameters.
-face_detection/dataset_loader.py: Loads the dataset and prepares it for training.
-face_detection/losses.py: Custom loss function used during training.
-face_detection/model.py: Model architecture.
-face_detection/utils.py: Helper functions .





