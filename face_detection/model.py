import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense

# --- Build model ---
def build_model():
    input_layer = Input(shape=(120, 120, 3))
    vgg = VGG16(include_top=False)(input_layer)
    f1 = GlobalMaxPooling2D()(vgg)

    # Classification head
    class_output = Dense(2048, activation='relu')(f1)
    class_output = Dense(1, activation='sigmoid', name='class_output')(class_output)

    # Regression head
    bbox_output = Dense(2048, activation='relu')(f1)
    bbox_output = Dense(4, name='bbox_output')(bbox_output)

    return Model(inputs=input_layer, outputs=[class_output, bbox_output])



# --- Optimizer function ---
def get_optimizer(batches_per_epoch):
    lr_decay = (1 / 0.75 - 1) / batches_per_epoch
    return tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)


# --- Custom training loop wrapper ---
class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localisationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localisationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch
        y_class, y_bbox = y
        y_class = tf.reshape(y_class, (-1, 1))
        y_bbox = tf.reshape(y_bbox, (-1, 4))
        y_class = tf.cast(y_class, tf.int32)
        y_bbox = tf.cast(y_bbox, tf.float32)

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_classloss = self.closs(y_class, classes)
            batch_localisationloss = self.lloss(y_bbox, coords)
            total_loss = batch_localisationloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {
            "total_loss": total_loss,
            "class_loss": batch_classloss,
            "regress_loss": batch_localisationloss
        }

    def test_step(self, batch, **kwargs):
        X, y = batch
        y_class, y_bbox = y
        y_class = tf.reshape(y_class, (-1, 1))
        y_bbox = tf.reshape(y_bbox, (-1, 4))
        y_class = tf.cast(y_class, tf.int32)
        y_bbox = tf.cast(y_bbox, tf.float32)

        classes, coords = self.model(X, training=False)
        batch_classloss = self.closs(y_class, classes)
        batch_localisationloss = self.lloss(y_bbox, coords)
        total_loss = batch_localisationloss + 0.5 * batch_classloss
        return {
            "total_loss": total_loss,
            "class_loss": batch_classloss,
            "regress_loss": batch_localisationloss
        }

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


