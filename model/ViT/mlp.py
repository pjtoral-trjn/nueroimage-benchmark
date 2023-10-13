import tensorflow as tf
from tensorflow import Variable, Module
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Dropout, Permute, LayerNormalization, Layer
from tensorflow.keras.activations import softmax, gelu


class MLP(Layer):
    """
    Multi Layer Perceptron

    Input Parameters
    ----------------
    input_features: int ; number of input features
    hidden_features: int ; number of nodes in the hidden layer
    out_features: int ; number of output features
    dropout_probability: float ; drop out probability

    Attribute
    ---------
    fc_1: Dense ; first linear layer
    act: activation
    fc_2: Dense ; second linear layer
    drop: Dropout ; drop out layer
    """

    def __init__(self, input_features, hidden_features, out_features, dropout_probability=0.):
        super(MLP, self).__init__()
        self.fc_1 = Dense(hidden_features)
        self.act = gelu
        self.fc_2 = Dense(out_features)
        self.drop = Dropout(dropout_probability)

    def call(self, x):
        x = self.fc_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc_2(x)
        x = self.drop(x)
        return x

    def get_config(self):
        return {"fc_1": self.fc_1, "act": self.act, "fc_2": self.fc_2, "drop": self.drop}
