import tensorflow as tf
from numpy.random import default_rng

def generate_image_noise(noise_shape):
    rng = default_rng()

    generator = tf.random.Generator.from_seed(rng.integers(10000))

    result = tf.random.stateless_uniform(
        shape=noise_shape,
        seed=(rng.integers(10000), rng.integers(10000)),
        minval=0,
        maxval=1
    )

    result = tf.math.scalar_mul(255, result)
    result = tf.cast(result, tf.int32)

    return tf.cast(result, tf.float32)