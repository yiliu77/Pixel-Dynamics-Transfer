import numpy as np
import gym
import cv2


class StackedEnv(gym.Wrapper):
    def __init__(self, env, width, height, n_img_stack, n_action_repeats):
        super(StackedEnv, self).__init__(env)
        self.width = width
        self.height = height
        self.n_img_stack = n_img_stack
        self.n_action_repeats = n_action_repeats
        self.stack = []
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = np.zeros([width, height, n_img_stack])

    def reset(self):
        img_rgb = super(StackedEnv, self).reset()
        img_gray = self.preprocess(img_rgb)
        self.stack = [img_gray] * self.n_img_stack
        return np.concatenate(self.stack, axis=2)

    def step(self, action):
        total_reward = 0
        done = False
        img_rgb = None
        for i in range(self.n_action_repeats):
            img_rgb, reward, done, info = super(StackedEnv, self).step(action)
            total_reward += reward
            if done:
                break
        img_gray = self.preprocess(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.n_img_stack
        return np.concatenate(self.stack, axis=2), total_reward, done, info

    def preprocess(self, rgb_img):
        bl_img = cv2.cvtColor(rgb_img.astype('float32'), cv2.COLOR_BGR2GRAY)
        res = cv2.resize(bl_img, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
        return res[:, :, np.newaxis]
