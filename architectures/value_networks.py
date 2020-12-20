import tensorflow as tf
from tensorflow import keras
from architectures.builder import Model
import numpy as np
from architectures.builder import separate_emb_layers


class ContQNet(keras.Model):
    def __init__(self, state_dim, action_dim, model_config):
        super().__init__()
        embedding_config, layers_config = separate_emb_layers(model_config)
        self.embedding = Model(embedding_config) if embedding_config is not None else None
        self.model = Model(layers_config)
        self([tf.constant(np.zeros(shape=[1] + list(state_dim), dtype=np.float32)),
              tf.constant(np.zeros(shape=(1, action_dim), dtype=np.float32))])

    def call(self, states_actions, training=True, mask=None):
        states, actions = states_actions
        states = states if self.embedding is None else self.embedding(states)
        return self.model(tf.concat([states, actions], axis=1), training)

    def get_config(self):
        raise NotImplementedError()


class ContTwinQNet(keras.Model):
    def __init__(self, state_dim, action_dim, model_config):
        super().__init__()
        embedding_config, layers_config = separate_emb_layers(model_config)
        # TODO
        self.embedding = Model(embedding_config) if embedding_config is not None else None
        self.q_net1 = Model(layers_config)
        self.q_net2 = Model(layers_config)

        self([tf.constant(np.zeros(shape=[1] + list(state_dim), dtype=np.float32)),
              tf.constant(np.zeros(shape=(1, action_dim), dtype=np.float32))])

    def call(self, states_actions, training=True, mask=None):
        states, actions = states_actions
        states = states if self.embedding is None else self.embedding(states)
        states_actions = tf.concat([states, actions], axis=1)
        q1_out, q2_out = self.q_net1(states_actions, training), self.q_net2(states_actions, training)
        return tf.math.minimum(q1_out, q2_out), q1_out, q2_out

    def get_config(self):
        raise NotImplementedError()
