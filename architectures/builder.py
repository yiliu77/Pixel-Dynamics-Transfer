import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import copy
import numpy as np


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad

@tf.custom_gradient
def grad_reverse_small(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -0.05 * dy

    return y, custom_grad


def polyak_update(network, target_network, tau):
    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable)
        else:
            return target_variable.assign(tau * source_variable + (1.0 - tau) * target_variable)

    update_ops = [update_op(target_var, source_var, tau) for target_var, source_var in
                  zip(target_network, network)]
    return tf.group(name="update_all_variables", *update_ops)


def gen_noise(tensor, scale):
    return tensor + tf.random.normal(tensor.shape, mean=0.0, stddev=scale)


def separate_emb_layers(model_config):
    model_config1, model_config2 = copy.deepcopy(model_config), copy.deepcopy(model_config)
    for i, layer in enumerate(model_config["architecture"]):
        if "mark_embedding" in layer["name"].lower():
            model_config1["architecture"] = model_config["architecture"][:i]
            model_config2["architecture"] = model_config["architecture"][i + 1:]
            return model_config1, model_config2
    return None, model_config2


# class StackedEmbedding(keras.Model):
#     def __init__(self, model_config, mode):
#         super().__init__()
#         self.mode = mode
#         self.model = Model(model_config)
#
#     def call(self, x, training=True, mask=None):
#         n_channels = x.shape[3]
#         outputs = []
#         if self.mode == "rgb":
#             for i in range(0, n_channels, 3):
#                 outputs.append(self.model(x[:, :, :, i: i + 3]))
#         else:
#             for i in range(0, n_channels):
#                 outputs.append(self.model(x[:, :, :, i]))
#         return tf.concat(outputs, 1)


class Model(keras.Model):
    def __init__(self, model_config):
        super().__init__()
        architecture, hidden_activation, output_activation = model_config["architecture"], model_config[
            "hidden_activation"], model_config["output_activation"]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.cnn_config, self.linear_config, self.split_config = Model._separate_config(architecture)
        self.cnn_layers = self._init_cnn_layers(self.cnn_config)
        self.linear_layers = self._init_linear_layers(self.linear_config)
        self.left_layers, self.right_layers = self._init_split_layers(self.split_config)

    @staticmethod
    def _separate_config(model_config):
        cnn_config, linear_config, split_config = [], [], []
        layer_type = "conv"

        for layer_config in model_config:
            layer_name = layer_config["name"].lower()
            if "conv" in layer_name:
                assert layer_type == "conv", "Conv layer configuration cannot be parsed correctly"
                cnn_config.append(layer_config)
            elif "linear" in layer_name:
                assert layer_type in ["conv", "linear"], "Linear layer configuration cannot be parsed correctly"
                layer_type = "linear"
                linear_config.append(layer_config)
            elif "split" in layer_name:
                assert layer_type in ["conv", "linear", "split"], "Split layer configuration cannot be parsed correctly"
                layer_type = "split"
                split_config.append(layer_config)
            else:
                "Model layer cannot be parsed correctly"
        return cnn_config, linear_config, split_config

    def _init_cnn_layers(self, cnn_config):
        cnn_layers = []
        initializer = tf.initializers.GlorotUniform()
        for i in range(len(cnn_config)):
            layer_dict = cnn_config[i]

            filters = layer_dict['channels']
            kernel_size = layer_dict['kernel_size'] if 'kernel_size' in layer_dict else 4
            stride = layer_dict['stride'] if 'stride' in layer_dict else 1
            padding = layer_dict['padding'] if 'padding' in layer_dict else "same"
            if i != len(cnn_config) - 1 or len(self.linear_config) != 0 or len(self.split_config) != 0:
                activation = self.hidden_activation
            else:
                activation = self.output_activation

            cnn_layers.append(layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
                                            activation=activation, kernel_initializer=initializer))
        return cnn_layers

    def _init_linear_layers(self, linear_config):
        linear_layers = []
        for i in range(len(linear_config)):
            size = linear_config[i]['size']
            if i != len(linear_config) - 1 or len(self.split_config) != 0:
                activation = self.hidden_activation
            else:
                activation = self.output_activation

            linear_layers.append(layers.Dense(size, activation=activation))
        return linear_layers

    def _init_split_layers(self, split_config):
        left_layers, right_layers = [], []
        for i in range(len(split_config)):
            sizes = split_config[i]['sizes']
            if i != len(split_config) - 1:
                activation = self.hidden_activation
            else:
                activation = self.output_activation

            left_layers.append(layers.Dense(sizes[0], activation=activation))
            right_layers.append(layers.Dense(sizes[1], activation=activation))
        return left_layers, right_layers

    def call(self, x, training=True, mask=None):
        if len(self.cnn_layers) != 0:
            for layer in self.cnn_layers:
                x = layer(x)
            x = layers.Flatten()(x)

        if len(self.linear_layers) != 0:
            for layer in self.linear_layers:
                x = layer(x)

        if len(self.left_layers) != 0 and len(self.right_layers) != 0:
            l, r = x, x
            for layer in self.left_layers:
                l = layer(l)
            for layer in self.right_layers:
                r = layer(r)
            x = [l, r]
        return x

    def get_config(self):
        raise NotImplementedError()
