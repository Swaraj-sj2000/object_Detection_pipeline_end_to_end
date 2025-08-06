import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import os
from face_detection.model import FaceTracker
from face_detection.losses import localisation_loss


def visualize_sample(image, label):
    class_id, bbox = label
    img = image.numpy()
    bbox = bbox.numpy()

    h, w = img.shape[:2]
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)

    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                      fill=False, edgecolor='red', linewidth=2))
    plt.title(f'Class: {class_id[0].numpy()}')
    plt.show()

def draw_bbox(image, bbox, label=None):
    x, y, w, h = [int(i) for i in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if label is not None:
        cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36,255,12), 2)
    return image

def save_model(model, path='model.h5'):
    model.save(path)

def load_trained_model(path='FaceTracker.h5'):
    return tf.keras.models.load_model(path, custom_objects={
        'FaceTracker': FaceTracker,
        'localisation_loss': localisation_loss
    })


def plot_training_history(h1):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].plot(h1['total_loss'], 'b-')
    axes[0, 0].set_title("Train Total Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(['Train'])

    axes[0, 1].plot(h1['class_loss'], 'b-')
    axes[0, 1].set_title("Train Classification Loss")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(['Train'])

    axes[0, 2].plot(h1['regress_loss'], 'b-')
    axes[0, 2].set_title("Train BBox Loss")
    axes[0, 2].set_xlabel("Epochs")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].legend(['Train'])

    axes[1, 0].plot(h1['val_total_loss'], 'r--')
    axes[1, 0].set_title("Val Total Loss")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend(['Val'])

    axes[1, 1].plot(h1['val_class_loss'], 'r--')
    axes[1, 1].set_title("Val Classification Loss")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend(['Val'])

    axes[1, 2].plot(h1['val_regress_loss'], 'r--')
    axes[1, 2].set_title("Val BBox Loss")
    axes[1, 2].set_xlabel("Epochs")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].legend(['Val'])

    plt.tight_layout()
    plot_path="documents"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plt.savefig(os.path.join(plot_path, "training_plot.png"))

    print(f"Training history saved to {plot_path}")
    plt.show()
