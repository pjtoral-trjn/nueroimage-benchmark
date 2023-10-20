# 3D VGG16
import tensorflow as tf
import tensorflow_addons as tfa

class VGG16:
  def __init__(self, args):
    self.args = args

  def get_model(self):

    def vgg_block(inputs, num_filter, name, block_num = 0):
        if block_num <= 2:
            inputs = tf.keras.layers.Conv3D(num_filter, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.Conv3D(num_filter, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
        else:
            inputs = tf.keras.layers.Conv3D(num_filter, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.Conv3D(num_filter, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.Conv3D(num_filter, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)


    images = tf.keras.Input((96, 96, 96, 1))

    ## First layer
    inputs = vgg_block(images, 32, "vgg16_conv_block1", 1)

    ## Second layer
    inputs = vgg_block(images, 64, "vgg16_conv_block2", 2)

    ## Third layer
    inputs = vgg_block(images, 128, "vgg16_conv_block3", 3)

    ## Fourth layer
    inputs = vgg_block(images, 256, "vgg16_conv_block4", 4)

    ## Fifth layer
    inputs = vgg_block(images, 256, "vgg16_conv_block5", 5)

    # Last Layer
    outputs = tf.keras.layers.Flatten(name="flatten")(inputs)
    outputs = tf.keras.layers.Dense(units=4096, name="fully_connected_1", activation="relu")(outputs)
    outputs = tf.keras.layers.Dense(units=4096, name="fully_connected_2", activation="relu")(outputs)

    outputs = tf.keras.layers.Dense(units=1, name="Cognitive-Assessment-3DVGG16", activation="softmax")(outputs)

    # Define the model
    return tf.keras.Model(images, outputs, name="3DVGG16")