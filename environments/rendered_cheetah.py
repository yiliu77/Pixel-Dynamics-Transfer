import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import GlfwContext


class CheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, path):
        self.n_steps = 0
        self._max_episode_steps = 500
        GlfwContext(offscreen=True)

        mujoco_env.MujocoEnv.__init__(self, path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        self.n_steps += 1
        done = self.n_steps >= self._max_episode_steps
        target = self.sim.data.qpos.flat.copy()[1:]
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, target=target)

    def _get_obs(self):
        return self.render(mode='rgb_array', width=100, height=100)[40:, :, :] / 255

    def reset_model(self):
        self.n_steps = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.elevation = -20
        self.viewer.cam.trackbodyid = 0