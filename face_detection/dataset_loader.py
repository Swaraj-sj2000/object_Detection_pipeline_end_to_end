import tensorflow as tf
import json
import os
import numpy as np

def load_image(image_path, img_size=(120, 120)):
    byte_img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(byte_img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0,1]
    return img

def load_label(label_path):
    '''Use py_function so we can work with regular Python inside TF pipeline'''
    def _parse_label(path):
        with open(path.numpy().decode('utf-8'), 'r', encoding='utf-8') as f:
            label = json.load(f)
        return [label['class']], label['bbox']

    return tf.py_function(_parse_label, [label_path], [tf.uint8, tf.float32])

def build_dataset(image_dir, label_dir, img_size=(120, 120), batch_size=8, shuffle_buffer=5000, prefetch_size=tf.data.AUTOTUNE):
    '''Prepare image and label datasets'''
    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, '*.jpg'), shuffle=False)
    label_files = tf.data.Dataset.list_files(os.path.join(label_dir, '*.json'), shuffle=False)

    images = image_files.map(lambda x: load_image(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    labels = label_files.map(load_label, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset





