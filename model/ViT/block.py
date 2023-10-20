import tensorflow as tf
from tensorflow import Variable, Module
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, Permute, LayerNormalization, Layer
from tensorflow.keras.activations import softmax, gelu
from model.ViT.mlp import MLP
from model.ViT.attention import Attention

class Block(Layer):
    """
    Transformer Block

    Input Parameters
    ----------------
    embed_dimension: int ; embedding dimension
    n_heads: int ; number of attention heads
    mlp_ratio: float ; hidden dimension size wrt embed_dimension
    qkv_bias: bool ; include bias to q,k,v projections
    dropout_1, dropout_2: float ; dropout probabilities

    Attribute
    ---------
    norm_1, norm_2: LayerNormalization ; Layer normalization
    attn: Attention ; Attention Module
    mlp: MLP ; MLP module
    """

    def __init__(self, embed_dimension=768, n_heads=12, mlp_ratio=4.0, qkv_bias=True, dropout_1=0., dropout_2=0.):
        super(Block, self).__init__()
        self.norm_1 = LayerNormalization(epsilon=1e-6)
        self.norm_2 = LayerNormalization(epsilon=1e-6)
        self.attn = Attention(embed_dimension, n_heads, qkv_bias)
        hidden_features = (embed_dimension * mlp_ratio)
        self.mlp = MLP(input_features=embed_dimension, hidden_features=hidden_features, out_features=embed_dimension)

    def call(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x

    def get_config(self):
        return {"norm_1": self.norm_1, "norm_2": self.norm_2, "attn": self.attn, "mlp": self.mlp}