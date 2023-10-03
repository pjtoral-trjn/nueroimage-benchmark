import tensorflow as tf
import tensorflow_addons as tfa
class TCNN(tf.keras.Model):
    def convolution_block(self, inputs, num_filter, name):
        inputs = tf.keras.layers.Conv3D(num_filter, 3, strides=1, padding="same")(inputs)
        inputs = tfa.layers.InstanceNormalization(center=False, scale=False)(inputs, training=True)
        inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid")(inputs)
        inputs = tf.nn.relu(inputs)

        return inputs


    def __init__(self, parser):
        super().__init__()
        images = tf.keras.Input((96, 96, 96, 1))
        self.cb_1 = self.convolution_block(images, 32, "conv_block1")
        self.cb_2 = self.convolution_block(self.cb_1, 64, "conv_block2")
        self.cb_3 = self.convolution_block(self.cb_2, 128, "conv_block3")
        # inputs = convolution_block(images, 256, "conv_block4")

        # Last Layer
        self.cnv_1 = tf.keras.layers.Conv3D(64, 1, strides=1, name="post_conv")(self.cb_3)
        self.inst_norm = tfa.layers.InstanceNormalization(center=False, scale=False)(self.cnv_1, training=True)
        self.relu = tf.nn.relu(self.inst_norm, name="post_relu")
        self.avg_pool = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2), strides=2, name="post_avg_pool")(self.relu)

        self.dropout_lyr = tf.keras.layers.Dropout(rate=self.drop_out, name="drop")(self.avg_pool)

        self.cnv_2 = tf.keras.layers.Conv3D(64, 1, strides=1, name="reg_conv")(self.dropout_lyr)
        self.flatten_lyr = tf.keras.layers.Flatten(name="flatten")(self.cnv_2)

        self.output_lyr = tf.keras.layers.Dense(units=1, name="Cognitive-Assessment-3DCNN",
                                        # bias_initializer=tf.keras.initializers.RandomNormal(
                                        #     mean=np.mean(self.data.train_df[self.target_column]),
                                        #     stddev=np.std(self.data.train_df[self.target_column]),
                                        #     seed=self.seed_num)
                                        # activation="sigmoid"
                                        )(self.flatten_lyr)

        # Define the model
        # model = tf.keras.Model(images, self.output_lyr, name="3DCNN")
        # self.model = model
    def call(self, inputs):
        x = self.cb_1(inputs)
        x = self.cb_2(x)
        x = self.cb_3(x)
        x = self.cnv_1(x)
        x = self.inst_norm(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.dropout_lyr(x)
        x = self.cnv_2(x)
        x = self.flatten_lyr(x)

        return self.output_lyr(x)
