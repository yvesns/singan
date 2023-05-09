import tensorflow as tf
import utils as utils
from tensorflow.keras import layers
from numpy.random import default_rng

class Generator(tf.keras.Model):
    def __init__(self, kernel_count=32):
        super(Generator, self).__init__()

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
            3, 
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

        self.leaky_relu1 = layers.LeakyReLU()
        self.leaky_relu2 = layers.LeakyReLU()
        self.leaky_relu3 = layers.LeakyReLU()
        self.leaky_relu4 = layers.LeakyReLU()
        self.leaky_relu5 = layers.LeakyReLU()

    def execute_conv_net(self, net_input):
        result = self.conv2D1(net_input)
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

        return self.leaky_relu5(result)
    
    def call(self, noise_shape, image=None, training=False):
        result = utils.generate_image_noise(noise_shape)

        if (image != None):
            result = tf.math.add(result, image)

        result = self.execute_conv_net(result)

        if (image != None):
            result = tf.math.add(result, image)

        return result

    def get_loss(self, discriminator_result):
        return -tf.reduce_mean(discriminator_result)

    def get_reconstruction_loss(self, image, reconstructed_image=None, noise=None):
        new_reconstructed_image = None
        loss_function = tf.keras.losses.MeanSquaredError()

        if (reconstructed_image == None):
            new_reconstructed_image = self.execute_conv_net(noise)
        else:
            new_reconstructed_image = self.execute_conv_net(reconstructed_image)

        # print(new_reconstructed_image)
        # print(image)

        return {
            "reconstructed_image": new_reconstructed_image,
            "loss": loss_function(image, new_reconstructed_image)
        }