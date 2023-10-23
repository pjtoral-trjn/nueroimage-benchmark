import tensorflow as tf
import tensorflow_addons as tfa
class TCNN:
    def __init__(self, args, train_mean, train_std):
        self.args = args
        self.train_mean = train_mean
        self.train_std = train_std

    def get_model(self):
        # Tamgohna model
        def convolution_block(inputs, num_filter, name):
            inputs = tf.keras.layers.Conv3D(num_filter, 3, strides=1, padding="same")(inputs)
            inputs = tfa.layers.InstanceNormalization(center=False, scale=False)(inputs, training=True)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
            inputs = tf.nn.relu(inputs)

            return inputs

        images = tf.keras.Input((91, 109, 91, 1))
        inputs = convolution_block(images, 32, "conv_block1")
        inputs = convolution_block(inputs, 64, "conv_block2")
        inputs = convolution_block(inputs, 128, "conv_block3")
        # inputs = convolution_block(images, 256, "conv_block4")

        # Last Layer
        inputs = tf.keras.layers.Conv3D(64, 1, strides=1, name="post_conv")(inputs)
        inputs = tfa.layers.InstanceNormalization(center=False, scale=False)(inputs, training=True)
        inputs = tf.nn.relu(inputs, name="post_relu")
        inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2), strides=2, name="post_avg_pool")(inputs)

        inputs = tf.keras.layers.Dropout(rate=self.args.drop_out, name="drop")(inputs)

        outputs = tf.keras.layers.Conv3D(64, 1, strides=1, name="reg_conv")(inputs)
        outputs = tf.keras.layers.Flatten(name="flatten")(outputs)

        outputs = tf.keras.layers.Dense(units=1, name="Cognitive-Assessment-3DCNN",
                                        bias_initializer=tf.keras.initializers.RandomNormal(
                                            mean= self.train_mean,
                                            stddev=self.train_std,
                                            seed=5)
                                        # activation="sigmoid"
                                        )(outputs)

        # Define the model
        return tf.keras.Model(images, outputs, name="3DCNN")