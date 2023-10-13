import tensorflow as tf
from tensorflow import Variable, Module
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, Permute, LayerNormalization, Layer
from tensorflow.keras.activations import softmax, gelu


class Attention(Layer):
    """
    Attension Mechanism

    Input Parameters
    ----------------
    dimension: int ; input and output dimension of per token features
    n_heads: int ; number of attention heads
    qkv_bias: bool ; include bias to the q,k,v projections
    attn_drop: float; drop out applied to q,k,v tensors
    out_drop: float; drop out applied to the output tensor

    Attributes
    ----------
    scale: float ; normalizing constant for the dot product
    qkv: Dense ; Linear projections for q,k,v
    projection: Dense ; Linear mapping that takes concatenated output of all attention heads and maps
        it into a new space
    attn_dropout, out_dropout: Dropout; drop out layers
    """

    def __init__(self, dimension=768, n_heads=12, qkv_bias=True, attn_drop=0., out_drop=0.):
        super(Attention, self).__init__()
        self.dimension = dimension
        self.n_heads = n_heads
        self.head_dim = (dimension // n_heads)
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.out_drop = out_drop

        # *3?
        self.qkv = Dense(dimension * 3, use_bias=qkv_bias)
        self.projection = Dense(dimension)
        self.attn_dropout = Dropout(attn_drop)
        self.out_dropout = Dropout(out_drop)
        self.scale = self.head_dim ** -0.5

    def call(self, x):
        n_samples, n_tokens, dimension = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [n_samples, n_tokens, 3, self.n_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # qkv, n_samples, n_heads,n_ n_heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = tf.transpose(k, perm=[0, 1, 3, 2])
        dot_prod = (q @ k_t) * self.scale

        attn = softmax(dot_prod, axis=-1)
        attn = self.attn_dropout(attn)

        weighted_avg = attn @ v
        weighted_avg = tf.transpose(weighted_avg, perm=[0, 2, 1, 3])
        weighted_avg = tf.reshape(weighted_avg, [weighted_avg.shape[0], weighted_avg.shape[1],
                                                 weighted_avg.shape[2] * weighted_avg.shape[3]])

        x = self.projection(weighted_avg)
        x = self.out_dropout(x)
        return x

    def get_config(self):
        return {"dimension": self.dimension, "n_heads": self.n_heads, "head_dim": self.head_dim,
                "qkv_bias": self.qkv_bias, "attn_drop": self.attn_drop, "out_drop": self.out_drop, "qkv": self.qkv,
                "projection": self.projection, "attn_dropout": self.attn_dropout,
                "out_dropout": self.out_dropout, "scale": self.scale}
