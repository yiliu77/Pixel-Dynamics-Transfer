import tensorflow as tf
from tensorflow import keras
from architectures.builder import Model
import numpy as np
from architectures.builder import separate_emb_layers, gen_noise


class SAClassifier(keras.Model):
    def __init__(self, state_dim, action_dim, model_config, noise_scale=0.4):
        super().__init__()
        embedding_config, layers_config = separate_emb_layers(model_config)
        # TODO
        self.embedding = Model(embedding_config) if embedding_config is not None else None
        self.classifier = Model(layers_config)
        self.noise_scale = noise_scale

        self([tf.constant(np.zeros(shape=[1] + list(state_dim), dtype=np.float32)),
              tf.constant(np.zeros(shape=(1, action_dim), dtype=np.float32))])

    def call(self, states_actions, training=True, mask=None):
        states, actions = states_actions
        states = states if self.embedding is None else tf.stop_gradient(self.embedding(states))
        states_actions = tf.concat([states, actions], axis=1)
        return self.classifier(states_actions + gen_noise(self.noise_scale, states_actions))

    def get_config(self):
        raise NotImplementedError()


class SASClassifier(keras.Model):
    def __init__(self, state_dim, action_dim, model_config, noise_scale=0.4):
        super().__init__()
        embedding_config, layers_config = separate_emb_layers(model_config)
        self.embedding = Model(embedding_config) if embedding_config is not None else None
        self.classifier = Model(layers_config)
        self.noise_scale = noise_scale

        self([tf.constant(np.zeros(shape=[1] + list(state_dim), dtype=np.float32)),
              tf.constant(np.zeros(shape=(1, action_dim), dtype=np.float32)),
              tf.constant(np.zeros(shape=[1] + list(state_dim), dtype=np.float32))])

    def call(self, states_actions_states, training=True, mask=None):
        states, actions, next_states = states_actions_states
        if self.embedding is not None:
            states = tf.stop_gradient(self.embedding(states))
            next_states = tf.stop_gradient(self.embedding(next_states))
        states_actions_states = tf.concat([states, actions, next_states], axis=1)
        return self.classifier(states_actions_states + gen_noise(self.noise_scale, states_actions_states))

    def get_config(self):
        raise NotImplementedError()