from matplotlib import pyplot as plt
from environments.stacked import StackedEnv
from environments.rendered_cheetah import CheetahEnv
import gym
from environments.broken_joint import BrokenJointEnv


env = StackedEnv(BrokenJointEnv(CheetahEnv(
    '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah.xml'), [1]), 20, 50,
    4, 2)
# env.model.opt.gravity[-1] = -20
while True:
    state = env.reset()
    env.reset()
    done = False
    total_reward = 0
    render_img = plt.imshow(state[:, :, 0])
    while not done:
        # env.render()
        env.step(env.action_space.sample())
        next_state, reward, done, info = env.step(env.action_space.sample())
        plt.imshow(state[:, :, 1], cmap='gray')
        plt.show()
        print(done)
        total_reward += reward
        state = next_state
        if done:
            break
