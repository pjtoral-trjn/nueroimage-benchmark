# ResBlocks
import tensorflow as tf
import tensorflow_addons as tfa

class RESNET3D:
  def __init__(self, args, train_mean, train_std):
    self.args = args
    self.train_mean = train_mean
    self.train_std = train_std


  def get_model(self, classification_transfer_learning=False):

    def convolution_block(inputs, num_filter, name):
            inputs = tf.keras.layers.Conv3D(num_filter, 3, strides=1, padding="same")(inputs)
            inputs = tfa.layers.InstanceNormalization(center=False, scale=False)(inputs, training=True)
            # inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
            # inputs = tf.nn.relu(inputs)

            return inputs

    images = tf.keras.Input((96, 96, 96, 1))
    input_skip = images

    ## First layer
    inputs = convolution_block(images, 32, "resnet_conv_block1")
    inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
    inputs = tf.nn.relu(inputs)

    ## Second Layer
    inputs = convolution_block(inputs, 64, "resnet_conv_block2")
    inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
    inputs = tf.nn.relu(inputs)

    # Prcoessing Residue
    input_skip_block = convolution_block(input_skip, 64, "input_skip_block")
    input_skip_block = tf.nn.relu(input_skip_block)

    # Add Residue
    inputs = tf.keras.layers.Add()([inputs, input_skip_block])
    inputs = tf.nn.relu(inputs)

    # ## Third layer
    inputs = convolution_block(inputs, 128, "resnet_conv_block3")
    inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
    inputs = tf.nn.relu(inputs)

    ## Fourth Layer
    inputs = convolution_block(inputs, 256, "resnet_conv_block4")
    inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
    inputs = tf.nn.relu(inputs)

    # Prcoessing Residue
    input_skip_block = convolution_block(input_skip_block, 256, "input_skip_block2")
    input_skip_block = tf.nn.relu(input_skip_block)

    # Add Residue
    inputs = tf.keras.layers.Add()([inputs, input_skip_block])
    inputs = tf.nn.relu(inputs)


    # Last Layer
    inputs = tf.keras.layers.Conv3D(64, 1, strides=1, name="post_resnet_conv")(inputs)
    inputs = tfa.layers.InstanceNormalization(center=False, scale=False)(inputs, training=True)
    inputs = tf.nn.relu(inputs, name="post_relu")
    inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2), strides=2, name="post_avg_pool")(inputs)

    inputs = tf.keras.layers.Dropout(rate=self.args.drop_out, name="drop")(inputs)

    outputs = tf.keras.layers.Conv3D(64, 1, strides=1, name="reg_conv")(inputs)
    outputs = tf.keras.layers.Flatten(name="flatten")(outputs)

    if not classification_transfer_learning:
        outputs = tf.keras.layers.Dense(units=1, name="Cognitive-Assessment-3DRSN",
                                            bias_initializer=tf.keras.initializers.RandomNormal(
                                                mean=self.train_mean,
                                                stddev=self.train_std,
                                                seed=5))(outputs)
    elif classification_transfer_learning:
        outputs = tf.keras.layers.Dense(units=1, name="Classification-3DRSN", activation="sigmoid")(outputs)

    # Define the model
    return tf.keras.Model(images, outputs, name="3DRSN")