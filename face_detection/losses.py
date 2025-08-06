import tensorflow as tf

def localisation_loss(y_true, y_hat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - y_hat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]
    h_pred = y_hat[:, 3] - y_hat[:, 1]
    w_pred = y_hat[:, 2] - y_hat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size

class_loss = tf.losses.BinaryCrossentropy()
regress_loss = localisation_loss
