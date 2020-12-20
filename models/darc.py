import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from architectures.builder import Model, gen_noise
from models.sac import ContSAC
from utils.replay_buffer import ReplayBuffer


class DARC(ContSAC):
    def __init__(self, policy_config, value_config, sa_config, sas_config, source_env, target_env,
                 log_dir="latest_runs", memory_size=1e5, warmup_games=10, batch_size=64, lr=0.0001, gamma=0.99,
                 tau=0.003, alpha=0.2, ent_adj=False, delta_r_scale=1.0, s_t_ratio=10, noise_scale=1.0,
                 target_update_interval=1, n_games_til_train=1, n_updates_per_train=1):
        super(DARC, self).__init__(policy_config, value_config, source_env, log_dir,
                                   memory_size, None, batch_size, lr, gamma, tau,
                                   alpha, ent_adj, target_update_interval, None, n_updates_per_train)
        self.delta_r_scale = delta_r_scale
        self.s_t_ratio = s_t_ratio
        self.noise_scale = noise_scale

        self.source_env = source_env
        self.target_env = target_env

        self.warmup_games = warmup_games
        self.n_games_til_train = n_games_til_train

        self.sa_classifier = Model(sa_config)
        self.sa_classifier_opt = Adam(learning_rate=lr)
        self.sas_adv_classifier = Model(sas_config)
        self.sas_adv_classifier_opt = Adam(learning_rate=lr)

        self.source_step = 0
        self.target_step = 0
        self.source_memory = self.memory
        self.target_memory = ReplayBuffer(self.memory_size, self.batch_size)

    @tf.function
    def train_step(self, s_states, s_actions, s_rewards, s_next_states, s_done_masks, *args):
        t_states, t_actions, _, t_next_states, _, game_count = args
        if len(s_rewards.shape) == 1:
            s_rewards = s_rewards[:, np.newaxis]
            s_done_masks = s_done_masks[:, np.newaxis]

        sa_inputs = tf.concat([s_states, s_actions], 1)
        sas_inputs = tf.concat([s_states, s_actions, s_next_states], 1)
        sa_logits = self.sa_classifier(sa_inputs + gen_noise(self.noise_scale, sa_inputs))
        sas_logits = self.sas_adv_classifier(sas_inputs + gen_noise(self.noise_scale, sas_inputs))
        sa_log_probs = tf.math.log(tf.nn.softmax(sa_logits) + 1e-12)
        sas_log_probs = tf.math.log(tf.nn.softmax(sas_logits + sa_logits) + 1e-12)

        delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
        if game_count >= 2 * self.warmup_games:
            s_rewards = tf.stop_gradient(s_rewards + self.delta_r_scale * tf.expand_dims(delta_r, 1))

        train_info = super(DARC, self).train_step(s_states, s_actions, s_rewards, s_next_states, s_done_masks)

        s_sa_inputs = tf.concat([s_states, s_actions], 1)
        s_sas_inputs = tf.concat([s_states, s_actions, s_next_states], 1)
        t_sa_inputs = tf.concat([t_states, t_actions], 1)
        t_sas_inputs = tf.concat([t_states, t_actions, t_next_states], 1)
        with tf.GradientTape(persistent=True) as tape:
            s_sa_logits = self.sa_classifier(s_sa_inputs + gen_noise(self.noise_scale, s_sa_inputs))
            s_sas_logits = self.sas_adv_classifier(
                s_sas_inputs + gen_noise(self.noise_scale, s_sas_inputs))
            t_sa_logits = self.sa_classifier(t_sa_inputs + gen_noise(self.noise_scale, t_sa_inputs))
            t_sas_logits = self.sas_adv_classifier(
                t_sas_inputs + gen_noise(self.noise_scale, t_sas_inputs))

            # TODO clean
            loss_function = tf.keras.losses.BinaryCrossentropy()
            label_zero = tf.concat([tf.ones((t_sa_logits.shape[0], 1), dtype=tf.int32),
                                    tf.zeros((t_sa_logits.shape[0], 1), dtype=tf.int32)], axis=1)
            label_one = tf.concat([tf.zeros((t_sa_logits.shape[0], 1), dtype=tf.int32),
                                   tf.ones((t_sa_logits.shape[0], 1), dtype=tf.int32)], axis=1)
            classify_loss = loss_function(label_zero, s_sa_logits)
            classify_loss += loss_function(label_one, t_sa_logits)
            classify_loss += loss_function(label_zero, s_sas_logits)
            classify_loss += loss_function(label_one, t_sas_logits)

        sa_classifier_grad = tape.gradient(classify_loss, self.sa_classifier.trainable_variables)
        self.sa_classifier_opt.apply_gradients(zip(sa_classifier_grad, self.sa_classifier.trainable_variables))
        sa_adv_classifier_grad = tape.gradient(classify_loss, self.sas_adv_classifier.trainable_variables)
        self.sas_adv_classifier_opt.apply_gradients(
            zip(sa_adv_classifier_grad, self.sas_adv_classifier.trainable_variables))

        s_sa_acc = tf.math.reduce_mean(1 - tf.dtypes.cast(tf.math.argmax(s_sa_logits, axis=1), tf.float32))
        s_sas_acc = tf.math.reduce_mean(1 - tf.dtypes.cast(tf.math.argmax(s_sas_logits, axis=1), tf.float32))
        t_sa_acc = tf.math.reduce_mean(tf.dtypes.cast(tf.math.argmax(t_sa_logits, axis=1), tf.float32))
        t_sas_acc = tf.math.reduce_mean(tf.dtypes.cast(tf.math.argmax(t_sas_logits, axis=1), tf.float32))

        train_info['Loss/Classify Loss'] = classify_loss
        train_info['Stats/Avg Delta Reward'] = tf.reduce_mean(delta_r)
        train_info['Stats/Avg Source SA Acc'] = s_sa_acc
        train_info['Stats/Avg Source SAS Acc'] = s_sas_acc
        train_info['Stats/Avg Target SA Acc'] = t_sa_acc
        train_info['Stats/Avg Target SAS Acc'] = t_sas_acc
        return train_info

    def train(self, num_games, deterministic=False):
        for i in range(num_games):
            source_reward, source_step = self.simulate_env(i, "source", deterministic)

            if i < self.warmup_games or i % self.s_t_ratio == 0:
                target_reward, target_step = self.simulate_env(i, "target", deterministic)
                # self.writer.add_scalar('Target Env/Rewards', target_reward, i)
                # self.writer.add_scalar('Target Env/N_Steps', target_step, i)
                print("TARGET: index: {}, steps: {}, total_rewards: {}".format(i, target_step, target_reward))

            if i >= self.warmup_games:
                # self.writer.add_scalar('Source Env/Rewards', source_reward, i)
                # self.writer.add_scalar('Source Env/N_Steps', source_step, i)
                if i % self.n_games_til_train == 0:
                    for _ in range(source_step * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s_s, s_a, s_r, s_s_, s_d = self.source_memory.sample()
                        t_s, t_a, t_r, t_s_, t_d = self.target_memory.sample()
                        train_info = self.train_step(s_s, s_a, s_r, s_s_, s_d, t_s, t_a, t_r, t_s_, t_d, i)
                        self.writer.add_train_step_info(train_info, i)
                    self.writer.write_train_step()
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

    # def save_model(self, folder_name):
    #     super(DARC, self).save_model(folder_name)
    #
    #     path = 'saved_weights/' + folder_name
    #     torch.save(self.sa_classifier.state_dict(), path + '/sa_classifier')
    #     torch.save(self.sas_adv_classifier.state_dict(), path + '/sas_adv_classifier')
    #
    # # Load model parameters
    # def load_model(self, folder_name, device):
    #     super(DARC, self).load_model(folder_name, device)
    #
    #     path = 'saved_weights/' + folder_name
    #     self.sa_classifier.load_state_dict(torch.load(path + '/sa_classifier', map_location=torch.device(device)))
    #     self.sas_adv_classifier.load_state_dict(
    #         torch.load(path + '/sas_adv_classifier', map_location=torch.device(device)))
