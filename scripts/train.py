import os
import tensorflow as tf
import face_detection as fd
from face_detection.config import *


def load_datasets():
    train = fd.build_dataset('aug_data/train/images', 'aug_data/train/labels', IMG_SIZE, BATCH_SIZE)
    val   = fd.build_dataset('aug_data/val/images', 'aug_data/val/labels', IMG_SIZE, BATCH_SIZE)
    test  = fd.build_dataset('aug_data/test/images', 'aug_data/test/labels', IMG_SIZE, BATCH_SIZE)
    return train, val, test

def get_callbacks():
    logdir = os.path.join('documents', 'logs')
    os.makedirs(logdir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    return [tensorboard_callback]

def train_model():
    
    # Load data
    train, val, test = load_datasets()

    # Build model
    model = fd.FaceTracker(fd.build_model())
    model.compile(
        opt=fd.get_optimizer(len(train)),
        classloss=fd.class_loss,
        localisationloss=fd.regress_loss
    )

    # Callbacks
    callbacks = get_callbacks()

    # Train
    history = model.fit(train, epochs=EPOCHS, validation_data=val, callbacks=callbacks)

    # Plot & Save
    fd.plot_training_history(history, save_path='documents/training_plot.png')
    model.save('FaceTracker.h5')

    return model, history

# Entry point for script execution
if __name__ == "__main__":
    train_model()
