import os

import numpy as np
import tensorflow as tf
import time
from tensorflow import losses
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt
from architectures.gaussian_policy import ContGaussianPolicy
from architectures.builder import polyak_update
from architectures.value_networks import ContTwinQNet
from utils.replay_buffer import ReplayBuffer
from utils.tensor_writer import TensorWriter


class ContSAC:
    def __init__(self, policy_config, value_config, env, log_dir="latest_runs",
                 memory_size=1e5, warmup_games=10, batch_size=64, lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2,
                 ent_adj=False, target_update_interval=1, n_games_til_train=1, n_updates_per_train=1):
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.log_dir = log_dir

        self.memory_size = memory_size
        self.warmup_games = warmup_games
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        self.action_range = (env.action_space.low, env.action_space.high)
        self.policy = ContGaussianPolicy(policy_config, state_dim, self.action_range)
        self.policy_opt = Adam(learning_rate=lr)

        self.twin_q = ContTwinQNet(state_dim, action_dim, value_config)
        self.twin_q_opt = Adam(learning_rate=lr)
        self.target_twin_q = ContTwinQNet(state_dim, action_dim, value_config)
        polyak_update(self.twin_q.weights, self.target_twin_q.weights, 1.0)

        self.tau = tau
        self.gamma = gamma
        self.n_until_target_update = target_update_interval
        self.n_games_til_train = n_games_til_train
        self.n_updates_per_train = n_updates_per_train

        self.alpha = alpha
        self.ent_adj = ent_adj
        if ent_adj:
            self.target_entropy = -float(len(self.action_range))
            self.log_alpha = tf.Variable(0, dtype=tf.float32)
            self.alpha = tf.Variable(0, dtype=tf.float32)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.alpha_opt = Adam(learning_rate=lr)
        self.total_train_steps = 0

    def get_action(self, state, deterministic=False):
        state = state[np.newaxis, :]
        if deterministic:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.numpy()[0]

    @tf.function
    def train_step(self, states, actions, rewards, next_states, done_masks):
        if len(rewards.shape) == 1:
            rewards = rewards[:, np.newaxis]
            done_masks = done_masks[:, np.newaxis]

        with tf.GradientTape(persistent=True) as tape:
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            next_q = self.target_twin_q([next_states, next_actions])[0]
            v = next_q - self.alpha * next_log_probs
            expected_q = tf.stop_gradient(rewards + done_masks * self.gamma * v)

            # Q backprop
            q_val, pred_q1, pred_q2 = self.twin_q([states, actions])
            q_loss = losses.mean_squared_error(pred_q1, expected_q) + losses.mean_squared_error(pred_q2, expected_q)
            q_loss = tf.reduce_mean(q_loss)

            # Policy backprop
            s_action, s_log_prob, _ = self.policy.sample(states)
            policy_loss = self.alpha * s_log_prob - self.twin_q([states, s_action])[0]
            policy_loss = tf.math.reduce_mean(policy_loss)

            if self.ent_adj:
                alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(s_log_prob + self.target_entropy))

        twin_q_grad = tape.gradient(q_loss, self.twin_q.trainable_variables)
        self.twin_q_opt.apply_gradients(zip(twin_q_grad, self.twin_q.trainable_variables))
        policy_grad = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_opt.apply_gradients(zip(policy_grad, self.policy.trainable_variables))

        if self.ent_adj:
            log_alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_opt.apply_gradients(zip(log_alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

        del tape

        if self.total_train_steps % self.n_until_target_update == 0:
            polyak_update(self.twin_q.weights, self.target_twin_q.weights, self.tau)

        return {'Loss/Policy Loss': policy_loss,
                'Loss/Q Loss': q_loss,
                'Stats/Avg Q Val': tf.reduce_mean(q_val),
                'Stats/Avg Q Next Val': tf.reduce_mean(next_q),
                'Stats/Avg Alpha': self.alpha}

    def train(self, num_games, deterministic=False):
        path = 'runs/' + self.log_dir + "/" + time.strftime("%d-%m-%Y_%H-%M-%S")
        writer = TensorWriter(path)

        for i in range(num_games):
            total_reward = 0
            n_steps = 0
            done = False
            state = self.env.reset()
            while not done:
                if i <= self.warmup_games:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state, deterministic)

                next_state, reward, done, _ = self.env.step(action)
                done_mask = 1.0 if n_steps == self.env._max_episode_steps - 1 else float(not done)

                self.memory.add(state, action, reward, next_state, done_mask)

                n_steps += 1
                total_reward += reward
                state = next_state

            if i >= self.warmup_games:
                tf.summary.trace_on()
                with writer.writer.as_default():
                    tf.summary.scalar('Env/Rewards', total_reward, step=i)
                    tf.summary.scalar('Env/N_Steps', n_steps, step=i)
                    if i % self.n_games_til_train == 0:
                        for _ in range(max(200, n_steps * self.n_updates_per_train)):
                            self.total_train_steps += 1
                            s, a, r, s_, d = self.memory.sample()
                            train_info = self.train_step(s, a, r, s_, d)
                            writer.add_train_step_info(train_info, i)
                        writer.write_train_step()
                        print("--------------------")
                    tf.summary.trace_export(
                        name="my_func_trace",
                        step=i)
                    writer.flush()

            print("index: {}, steps: {}, total_rewards: {}".format(i, n_steps, total_reward))

    def eval(self, num_games, render=True):
        for i in range(num_games):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    plt.imshow(state[:, :, -3:])
                    plt.show()
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            print(i, total_reward)

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        self.policy.save_weights(path + '/policy')
        self.twin_q.save_weights(path + '/twin_q_net')

    # Load model parameters
    def load_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        self.policy.load_weights(path + '/policy')
        self.twin_q.load_weights(path + '/twin_q_net')

        polyak_update(self.twin_q.weights, self.target_twin_q.weights, 1)

    def load_embedding(self, path):
        self.policy.embedding.load_weights(path)
        self.twin_q.embedding.load_weights(path)
