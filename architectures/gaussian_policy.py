import tensorflow as tf
from tensorflow import keras
from architectures.builder import Model
import tensorflow_probability as tfp
import numpy as np
from architectures.builder import separate_emb_layers


# n_states -> split: n_actions * 2 (none)
class ContGaussianPolicy(keras.Model):
    def __init__(self, model_config, state_dim, action_range):
        super(ContGaussianPolicy, self).__init__()
        embedding_config, layer_config = separate_emb_layers(model_config)
        # TODO
        self.embedding = Model(embedding_config) if embedding_config is not None else None
        self.model = Model(layer_config)

        action_low, action_high = action_range
        self.action_scale = tf.constant((action_high - action_low) / 2, dtype=tf.float32)
        self.action_bias = tf.constant((action_high + action_low) / 2, dtype=tf.float32)
        self(tf.constant(np.zeros(shape=tuple([1] + list(state_dim)), dtype=np.float32)))

    def call(self, states, training=True, mask=None):
        states = states if self.embedding is None else self.embedding(states)
        mu, log_std = self.model(states, training)
        log_std = tf.clip_by_value(log_std, -20, 2)
        return mu, log_std

    def sample(self, states):
        mus, log_stds = self.call(states)
        stds = tf.math.exp(log_stds)

        normal_dists = tfp.distributions.Normal(mus, stds)
        outputs = normal_dists.sample()
        tanh_outputs = tf.math.tanh(outputs)
        actions = self.action_scale * tanh_outputs + self.action_bias
        mean_actions = self.action_scale * tf.math.tanh(mus) + self.action_bias

        log_probs = normal_dists.log_prob(outputs)
        # https://arxiv.org/pdf/1801.01290.pdf appendix C
        log_probs -= tf.math.log(self.action_scale * (1 - tf.math.pow(tanh_outputs, 2)) + 1e-6)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return actions, log_probs, mean_actions

    def get_config(self):
        raise NotImplementedError()
