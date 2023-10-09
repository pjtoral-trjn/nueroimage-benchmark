import tensorflow as tf
from tensorflow import Variable, Module
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, Permute, LayerNormalization, Layer
from tensorflow.keras.activations import softmax, gelu
from patch_embedding import PatchEmbedding
from block import Block
def VisionTransformer(input_batch_size, train_mean, train_std, image_size=96, patch_size=16, input_channels=1,
                         n_classes=1, embed_dimension=768, depth=12, n_heads=12, mlp_ratio=4.0,
                         qkv_bias=True, dropout_1=0, dropout_2=0.):
    patch_embedding = PatchEmbedding()
    embed_dimension = embed_dimension
    regression_token = Variable(tf.zeros([input_batch_size, 1, embed_dimension]))
    positional_embedding = Variable(tf.zeros([1, 1 + patch_embedding.n_patches, embed_dimension]))
    positional_dropout = Dropout(dropout_1)
    blocks = Sequential()
    for _ in range(depth):
        blocks.add(Block())
    #         n_samples = x.shape[0]
    images = tf.keras.layers.Input(shape=(96, 96, 96, 1), batch_size=input_batch_size)

    x = PatchEmbedding()(images)

    x = tf.concat([regression_token, x], axis=1)
    x = x + positional_embedding
    x = Dropout(dropout_1)(x)
    x = blocks(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    regression_vector_final = x[:, 0]
    outputs = Dense(units=1, name="Cognitive-Assessment-3DCNN",
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        mean=train_mean,
                        stddev=train_std,
                        seed=5)
                    # activation="sigmoid"
                    )(regression_vector_final)
    return tf.keras.Model(inputs=images, outputs=outputs)