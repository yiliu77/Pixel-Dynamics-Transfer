import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import losses
from tensorflow.keras.optimizers import Adam

from architectures.builder import Model, gen_noise, polyak_update, grad_reverse
from architectures.gaussian_policy import ContGaussianPolicy
from architectures.value_networks import ContTwinQNet
from utils.replay_buffer import ReplayBuffer
from utils.tensor_writer import TensorWriter


class DoubleSAC:
    def __init__(self, policy_config, value_config, embedding_config, s_config, sa_config, sas_config, source_env,
                 target_env, log_dir="latest_runs", memory_size=1e4, warmup_games=10, batch_size=64, lr=0.0001,
                 gamma=0.99, tau=0.003, alpha=0.2, ent_adj=False, delta_r_scale=1.0, s_t_ratio=5, noise_scale=1.0,
                 target_update_interval=1, n_games_til_train=1, n_updates_per_train=1):
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        self.source_env = source_env
        self.target_env = target_env
        self.log_dir = log_dir

        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train

        self.source_step = 0
        self.target_step = 0
        self.total_train_steps = 0
        self.source_memory = ReplayBuffer(memory_size, batch_size)
        self.target_memory = ReplayBuffer(memory_size, batch_size)

        self.tau = tau
        self.gamma = gamma
        self.n_until_target_update = target_update_interval
        self.n_games_til_train = n_games_til_train
        self.n_updates_per_train = n_updates_per_train

        self.embedding = Model(embedding_config)

        state_dim = source_env.observation_space.shape
        action_dim = source_env.action_space.shape[0]
        embedding_dim = [32]  # TODO
        self.action_range = (source_env.action_space.low, source_env.action_space.high)
        self.policy = ContGaussianPolicy(policy_config, embedding_dim, self.action_range)
        self.policy_opt = Adam(learning_rate=lr)

        self.twin_q = ContTwinQNet(state_dim, action_dim, value_config)
        self.twin_q_opt = Adam(learning_rate=lr)
        self.target_twin_q = ContTwinQNet(state_dim, action_dim, value_config)
        polyak_update(self.twin_q.weights, self.target_twin_q.weights, 1.0)

        self.alpha = alpha
        self.ent_adj = ent_adj
        if ent_adj:
            self.target_entropy = -float(len(self.action_range))
            self.log_alpha = tf.Variable(0, dtype=tf.float32)
            self.alpha = tf.Variable(0, dtype=tf.float32)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.alpha_opt = Adam(learning_rate=lr)

    def get_embedding(self, state):
        state = state[np.newaxis, :]
        return self.embedding(state).numpy()[0]

    def get_action(self, state, deterministic=False):
        state = state[np.newaxis, :]
        if deterministic:
            _, _, action = self.policy.sample(self.embedding(state))
        else:
            action, _, _ = self.policy.sample(self.embedding(state))
        return action.numpy()[0]

    @tf.function
    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        t_states, t_actions, t_rewards, t_next_states, t_done_masks, game_count = args

        if len(s_rewards.shape) == 1:
            s_rewards = s_rewards[:, np.newaxis]
            s_done_masks = s_done_masks[:, np.newaxis]
            t_rewards = t_rewards[:, np.newaxis]
            t_done_masks = t_done_masks[:, np.newaxis]

        states = tf.concat([s_states, t_states], axis=0)
        next_states = tf.concat([s_next_states, t_next_states], axis=0)
        actions = tf.concat([s_actions, t_actions], axis=0)
        rewards = tf.concat([s_rewards, t_rewards], axis=0)
        done_masks = tf.concat([s_done_masks, t_done_masks], axis=0)

        with tf.GradientTape(persistent=True) as tape:
            states_embed = self.embedding(states)
            next_states_embed = self.embedding(next_states)

            next_actions, next_log_probs, _ = self.policy.sample(next_states_embed)
            next_q = self.target_twin_q([next_states, next_actions])[0]
            v = next_q - self.alpha * next_log_probs
            expected_q = tf.stop_gradient(rewards + done_masks * self.gamma * v)

            # Q backprop
            q_val, pred_q1, pred_q2 = self.twin_q([states, actions])
            q_loss = losses.mean_squared_error(pred_q1, expected_q) + losses.mean_squared_error(pred_q2, expected_q)
            q_loss = tf.reduce_mean(q_loss)

            # Policy backprop
            s_action, s_log_prob, _ = self.policy.sample(states_embed)
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

        if self.total_train_steps % self.n_until_target_update == 0:
            polyak_update(self.twin_q.weights, self.target_twin_q.weights, self.tau)

        del tape

        return {'Loss/Policy Loss': policy_loss,
                'Loss/Q Loss': q_loss,
                'Stats/Avg Q Val': tf.reduce_mean(q_val),
                'Stats/Avg Q Next Val': tf.reduce_mean(next_q),
                'Stats/Avg Alpha': self.alpha}

    def train(self, num_games, deterministic=False):
        path = 'runs/' + self.log_dir + "/" + time.strftime("%d-%m-%Y_%H-%M-%S")
        writer = TensorWriter(path)

        for i in range(num_games):
            tf.summary.trace_on()
            with writer.writer.as_default():
                source_reward, source_step = self.simulate_env(i, "source", deterministic)

                if i < self.warmup_games or i % self.s_t_ratio == 0:
                    target_reward, target_step = self.simulate_env(i, "target", deterministic)
                    tf.summary.scalar('Target Env/Rewards', target_reward, step=i)
                    tf.summary.scalar('Target Env/N_Steps', target_step, step=i)
                    print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))

                if i >= self.warmup_games:
                    tf.summary.scalar('Source Env/Rewards', source_reward, step=i)
                    tf.summary.scalar('Source Env/N_Steps', source_step, step=i)
                    if i % self.n_games_til_train == 0:
                        for _ in range(source_step * self.n_updates_per_train):
                            self.total_train_steps += 1
                            s_s, s_a, s_r, s_s_, s_d = self.source_memory.sample()
                            t_s, t_a, t_r, t_s_, t_d = self.target_memory.sample()
                            train_info = self.train_step(s_s, s_a, s_r, s_s_, s_d, t_s, t_a, t_r, t_s_, t_d, i)
                            writer.add_train_step_info(train_info, i)
                        writer.write_train_step()
                        print("--------------------")

                print("SOURCE: index: {}, steps: {}, total_rewards: {}".format(i, source_step, source_reward))

    def simulate_env(self, game_count, env_name, deterministic):
        if env_name == "source":
            env = self.source_env
            memory = self.source_memory
        elif env_name == "target":
            env = self.target_env
            memory = self.target_memory
        else:
            raise Exception("Env name not recognized")

        total_rewards = 0
        n_steps = 0
        done = False
        state = env.reset()
        while not done:
            if game_count <= self.warmup_games:
                action = env.action_space.sample()
            else:
                action = self.get_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            done_mask = 1.0 if n_steps == env._max_episode_steps - 1 else float(not done)

            memory.add(state, action, reward, next_state, done_mask)

            if env_name == "source":
                self.source_step += 1
            elif env_name == "target":
                self.target_step += 1
            n_steps += 1
            total_rewards += reward
            state = next_state
        return total_rewards, n_steps

    def eval(self, num_games):
        for i in range(num_games):
            state = self.source_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ = self.source_env.step(action)
                total_reward += reward
                state = next_state
            print(i, total_reward)

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        self.embedding.save_weights(path + '/embedding')
        self.policy.save_weights(path + '/policy')
        self.twin_q.save_weights(path + '/twin_q')
    # Load model parameters

    def load_model(self, folder_name):
        path = 'saved_weights/' + folder_name

        self.embedding.load_weights(path + '/embedding')
        self.policy.load_weights(path + '/policy')
        self.twin_q.load_weights(path + '/twin_q')
