import tensorflow as tf
from tensorflow.keras import layers
from numpy.random import default_rng

class Discriminator(tf.keras.Model):
    def __init__(self, kernel_count=32):
        super(Discriminator, self).__init__()

        kernel_shape = (3, 3)

        self.conv2D1 = layers.Conv2D(
            kernel_count, 
            kernel_shape, 
            strides=(1, 1), 
            padding='same',
            use_bias=False
        )

        self.conv2D2 = layers.Conv2D(
            kernel_count, 
            kernel_shape, 
            strides=(1, 1), 
            padding='same', 
            use_bias=False
        )

        self.conv2D3 = layers.Conv2D(
            kernel_count,
            kernel_shape, 
            strides=(1, 1), 
            padding='same', 
            use_bias=False
        )

        self.conv2D4 = layers.Conv2D(
            kernel_count, 
            kernel_shape, 
            strides=(1, 1), 
            padding='same', 
            use_bias=False
        )

        self.conv2D5 = layers.Conv2D(
            1, 
            kernel_shape, 
            strides=(1, 1), 
            padding='same', 
            use_bias=False
        )

        self.batch_normalization1 = layers.BatchNormalization()
        self.batch_normalization2 = layers.BatchNormalization()
        self.batch_normalization3 = layers.BatchNormalization()
        self.batch_normalization4 = layers.BatchNormalization()
        self.batch_normalization5 = layers.BatchNormalization()

        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.2)
        self.leaky_relu4 = layers.LeakyReLU(alpha=0.2)
        self.leaky_relu5 = layers.LeakyReLU(alpha=0.2)
    
    def call(self, original_image, generated_image, training=False):
        result = self.conv2D1(generated_image)
        result = self.batch_normalization1(result)
        result = self.leaky_relu1(result)

        result = self.conv2D2(result)
        result = self.batch_normalization2(result)
        result = self.leaky_relu2(result)

        result = self.conv2D3(result)
        result = self.batch_normalization3(result)
        result = self.leaky_relu3(result)

        result = self.conv2D4(result)
        result = self.batch_normalization4(result)
        result = self.leaky_relu4(result)

        result = self.conv2D5(result)
        result = self.batch_normalization5(result)
        result = self.leaky_relu5(result)
        
        return result