import tensorflow as tf


def normalize_noise(noise):
    return tf.divide(
        tf.subtract(
            noise,
            tf.reduce_min(noise)
        ),
        tf.subtract(
            tf.reduce_max(noise),
            tf.reduce_min(noise)
        )
    )
