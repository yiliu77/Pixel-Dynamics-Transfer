import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt
from architectures.builder import Model
from architectures.builder import grad_reverse
from utils.replay_buffer import ReplayBuffer
from utils.tensor_writer import TensorWriter


class Regression:
    def __init__(self, embed_config, reg_config, domain_config, source_env, target_env,
                 log_dir="latest_runs", memory_size=1e4, batch_size=32, lr=0.0001, s_t_ratio=1, n_updates_per_train=1):
        self.source_env = source_env
        self.target_env = target_env
        self.log_dir = log_dir

        self.embed_model = Model(embed_config)
        self.reg_model = Model(reg_config)
        self.domain_model = Model(domain_config)
        self.model_opt = Adam(learning_rate=lr)

        self.step = 0
        self.s_t_ratio = s_t_ratio
        self.n_updates_per_train = n_updates_per_train
        self.source_memory = ReplayBuffer(memory_size, batch_size)
        self.target_memory = ReplayBuffer(memory_size, batch_size)

    @tf.function
    def train_step(self, s_states, s_labels, t_states, t_labels, t2_states, t2_labels):
        comb_label = np.vstack([np.tile([1., 0.], [s_states.shape[0], 1]),
                                np.tile([0., 1.], [t_states.shape[0], 1])])
        comb_label = comb_label.astype('float32')
        print(s_states.shape)

        with tf.GradientTape() as tape:
            s_embedding = self.embed_model(s_states)
            t_embedding = self.embed_model(t_states)
            t_embedding2 = self.embed_model(t2_states)

            # this custom loss is for pushing embeddings further from each other based on how far the labels are
            # apart from each other. This helps training accuracies very dramatically but since it is custom
            # don't wanna worry about how it is interacting with domain loss
            diff_loss = -tf.math.multiply(tf.reduce_mean(tf.math.square(t_labels - t2_labels), 1),
                                          tf.reduce_mean(tf.math.square(
                                              tf.math.l2_normalize(t_embedding, 1) - tf.math.l2_normalize(t_embedding2,
                                                                                                         1)), 1))
            diff_loss = tf.reduce_mean(diff_loss)

            comb_embedding = tf.concat([s_embedding, t_embedding], 0)
            comb_labels = tf.concat([s_labels, t_labels], 0)

            # DARC with new embedding, readme, try stuff from conditional distribution paper, robust env

            mus = self.reg_model(comb_embedding)
            # log_stds = tf.clip_by_value(log_stds, -20, 2)
            # stds = tf.math.exp(log_stds)
            #
            # normal_dists = tfp.distributions.Normal(mus, stds)
            # outputs = normal_dists.sample()
            tanh_outputs = tf.math.tanh(mus)

            comb_domain = self.domain_model(grad_reverse(comb_embedding))
            domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=comb_domain,
                                                                  labels=comb_label)
            domain_loss = tf.reduce_mean(domain_loss)

            regression_loss = tf.keras.losses.mean_squared_error(tanh_outputs, comb_labels)
            regression_loss = tf.reduce_mean(regression_loss)

            total_loss = regression_loss + 0.01 * domain_loss + 0.5 * diff_loss

        train_vars = self.reg_model.trainable_variables + self.embed_model.trainable_variables + \
                     self.domain_model.trainable_variables
        grad = tape.gradient(total_loss, train_vars)
        self.model_opt.apply_gradients(zip(grad, train_vars))

        return {'Loss/Mean Squared Error Loss': regression_loss,
                'Loss/Domain Loss': domain_loss}

    def train(self, num_games):
        path = 'runs/' + self.log_dir + "/" + time.strftime("%d-%m-%Y_%H-%M-%S")
        if not os.path.exists(path):
            os.makedirs(path)
        writer = TensorWriter(path)

        for i in range(num_games):
            for _ in range(10):
                self.simulate_env("source")

            if i % self.s_t_ratio == 0:
                for _ in range(10):
                    self.simulate_env("target")
                # print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))

            if i >= 5:
                with writer.writer.as_default():
                    for _ in range(self.n_updates_per_train * 4):
                        s_states, s_labels = self.source_memory.sample()
                        t_states, t_labels = self.target_memory.sample()
                        t2_states, t2_labels = self.target_memory.sample()
                        train_info = self.train_step(s_states, s_labels, t_states, t_labels, t2_states, t2_labels)
                        writer.add_train_step_info(train_info, i)
                    writer.write_train_step()
                    # print("--------------------")
            print(i)
            # print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def simulate_env(self, env_name):
        if env_name == "source":
            env = self.source_env
            memory = self.source_memory
        elif env_name == "target":
            env = self.target_env
            memory = self.target_memory
        else:
            raise Exception("Env name not recognized")

        done = False
        env.reset()
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            target = info['target']

            memory.add(state, target)

    def eval(self, num_games):
        for _ in range(num_games):
            done = False
            self.target_env.reset()
            while not done:
                # self.target_env.render()
                action = self.target_env.action_space.sample()
                state, reward, done, info = self.target_env.step(action)
                plt.imshow(state)
                plt.show()
                target = info['target']
                print(self.reg_model(self.embed_model(np.expand_dims(state, 0))).numpy()[0], target)

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        self.embed_model.save_weights(path + '/embed')
        self.reg_model.save_weights(path + '/reg')
        self.domain_model.save_weights(path + '/domain')

    # Load model parameters
    def load_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        self.embed_model.load_weights(path + '/embed')
        self.reg_model.load_weights(path + '/reg')
        self.domain_model.load_weights(path + '/domain')
