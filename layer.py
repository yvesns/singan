import tensorflow as tf
import numpy as np
import utils as utils
from generator import Generator
from discriminator import Discriminator

class Layer():
    def __init__(self, scaling_factor, kernel_count, image, image_shape):
        self.scaling_factor = scaling_factor
        self.generator = Generator(kernel_count)
        self.discriminator = Discriminator(kernel_count)
        self.image_shape = image_shape

        # self.image_sizes = tf.gather(image_shape, [0, 1])
        # self.image_sizes = tf.cast(self.image_sizes, tf.float32)
        # self.image_sizes = tf.math.scalar_mul(1 / self.scaling_factor, self.image_sizes)
        # self.image_sizes = tf.round(self.image_sizes)
        # self.image_sizes = tf.cast(self.image_sizes, tf.int32)

        # self.image = tf.image.resize(image, self.image_sizes)
        # self.image = tf.expand_dims(self.image, 0)

        self.image = tf.keras.layers.AveragePooling2D()(image)
        self.noise_shape = tf.identity(self.image.shape)

        # self.noise_shape = self.image_sizes.numpy()
        # self.noise_shape = np.append(self.noise_shape, 3)
        # self.noise_shape = np.insert(self.noise_shape, 0, 1)
        # self.noise_shape = tf.convert_to_tensor(self.noise_shape)

    def call(self, training=False):
        generated_image = self.generator.call(self.noise_shape, self.image, training)
        result = self.discriminator.call(self.image, generated_image, training)

        return generated_image

    def get_reconstruction_loss(self, reconstructed_image):
        print(self.image)
        print(reconstructed_image)

        return self.generator.get_reconstruction_loss(self.image, reconstructed_image=reconstructed_image)

class BaseLayer(Layer):
    def __init__(self, scaling_factor, kernel_count, image, image_shape):
        super().__init__(scaling_factor, kernel_count, image, image_shape)

        self.reconstruction_noise = utils.generate_image_noise(self.noise_shape)

    def call(self, training=False):
        generated_image = self.generator.call(self.noise_shape, None, training)
        result = self.discriminator.call(self.image, generated_image, training)

        return generated_image

    def get_reconstruction_loss(self):
        return self.generator.get_reconstruction_loss(self.image, noise=self.reconstruction_noise)