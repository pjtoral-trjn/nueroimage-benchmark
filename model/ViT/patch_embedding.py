import tensorflow as tf
from tensorflow import Variable, Module
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, Permute, LayerNormalization, Layer
from tensorflow.keras.activations import softmax, gelu

class PatchEmbedding(Layer):
    """
    - Split input image into patches
    - Positional embed the patches

    Input Parameters
    ----------------
    args; command line argument
    input_size: (int, int, int) ; square matrix
    patch_size: int ; equally divisible by the square matrix
    input_channels: int ; number of input channels to the image
    embed_dimension: int

    Attributes
    ----------
    n_patches: int ; number of patches inside our image
    projection: tf.keras.layers.Conv3d; convolution layer that splits into patches and includes embedding
    """
    def __init__(self, input_size=(96, 96, 96), patch_size=16, input_channels=1, embed_dimension=768):
        super(PatchEmbedding, self).__init__()
        self.input_size = input_size
        self.in_w, self.in_h, self.in_d = self.input_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embed_dimension = embed_dimension
        self.n_patches = (self.in_w // patch_size) ** 3

        self.projection = Conv3D(
            embed_dimension,
            patch_size,
            strides=patch_size,
            input_shape=(self.in_w, self.in_h, self.in_d, 1)
        )

    def call(self, x, training=False):
        tf.print('\n Input shape -->', tf.shape(x), '\n')
        x = self.projection(x)
        tf.print('After Convolution -->', tf.shape(x), '\n')
        tf.print('x.shape[0] -->', str(x.shape[0]), '\n')
        tf.print('self.embed_dimension -->', str(self.embed_dimension), '\n')
        tf.print('self.n_patches -->', str(self.n_patches), '\n')
        x = tf.reshape(x, [x.shape[0], self.n_patches, self.embed_dimension] ) # (n_samples, n_patches, embed_dimensions)
        return x