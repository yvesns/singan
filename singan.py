import tensorflow as tf
import math
from layer import Layer
from layer import BaseLayer

class SinGAN:
    def __init__(self, image, base_factor=2, layer_count=5):
        self.base_factor = base_factor
        self.layer_count = layer_count
        self.layers = []
        self.image = image
        
        kernel_count = 32
        kernel_count_scaling_factor = 1
        image_shape = tf.shape(image)

        self.layers.append(
            BaseLayer(
                pow(base_factor, layer_count - 1),
                kernel_count * math.ceil(layer_count / 4),
                image,
                image_shape
            )
        )

        for l in range(1, layer_count):
            kernel_count_scaling_factor = math.ceil((l + 1) / 4)

            self.layers.append(
                Layer(
                    pow(base_factor, layer_count - 1 - l),
                    kernel_count * kernel_count_scaling_factor,
                    image,
                    image_shape
                )
            )

    def call(self, training=False):
        generated_image = self.layers[0].call()

        for l in range(1, self.layer_count):
            generated_image = tf.keras.layers.UpSampling2D(size=[self.base_factor, self.base_factor])(generated_image)
            generated_image = self.layers[l].call(training)

        return tf.squeeze(generated_image)

    def get_reconstruction_loss(self):
        result = self.layers[0].get_reconstruction_loss()
        # print(result["reconstructed_image"])
        # test = tf.identity(result["reconstructed_image"])
        # print(test)
        # test = tf.keras.layers.UpSampling2D(size=[self.base_factor, self.base_factor])(test)
        result["reconstructed_image"] = tf.keras.layers.UpSampling2D(size=[self.base_factor, self.base_factor])(result["reconstructed_image"])
        # print(result["reconstructed_image"])
        # print(self.base_factor)
        # print(test)

        for l in range(1, self.layer_count):
            print("Iteration " + str(l))
            result = self.layers[l].get_reconstruction_loss(result["reconstructed_image"])
            result["reconstructed_image"] = tf.keras.layers.UpSampling2D(size=[self.base_factor, self.base_factor])(result["reconstructed_image"])

        return result