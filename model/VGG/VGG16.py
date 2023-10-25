import tensorflow as tf
class VGG16_3D:
    def __init__(self, args, train_mean, train_std):
        self.args = args
        self.train_mean = train_mean
        self.train_std = train_std

    def get_model(self):
        def convolution_block(inputs, num_filters):
            inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", activation="relu")(inputs)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
            return inputs

        images = tf.keras.Input((96, 96, 96, 1))

        # Block 1
        x = convolution_block(images, 64)

        # Block 2
        x = convolution_block(x, 128)

        # Block 3
        x = convolution_block(x, 256)
        x = tf.keras.layers.Conv3D(256, 3, strides=1, padding="same", activation="relu")(x)

        # Block 4
        x = convolution_block(x, 512)
        x = tf.keras.layers.Conv3D(512, 3, strides=1, padding="same", activation="relu")(x)

        # Block 5
        x = convolution_block(x, 512)
        x = tf.keras.layers.Conv3D(512, 3, strides=1, padding="same", activation="relu")(x)

        # Final layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(rate=self.args.drop_out)(x)
        outputs = tf.keras.layers.Dense(units=1,
                                        bias_initializer=tf.keras.initializers.RandomNormal(
                                            mean=self.train_mean,
                                            stddev=self.train_std,
                                            seed=5))(x)

        return tf.keras.Model(images, outputs, name="3D_VGG16")