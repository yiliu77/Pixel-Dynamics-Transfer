import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import GlfwContext


class RenderedReacher(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reacher_path):
        self.n_steps = 0
        self._max_episode_steps = 200
        GlfwContext(offscreen=True)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, reacher_path, 1)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        theta = self.sim.data.qpos.flat[:2]
        target = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:]])  # np.array(self.get_body_com("target")[:2] * 5)
        done = self.n_steps >= self._max_episode_steps
        self.n_steps += 1
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, target=target)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.6
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        self.n_steps = 0
        return self._get_obs()

    def _get_obs(self):
        return self.render(mode='rgb_array', width=100, height=100) / 255
