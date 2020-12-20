import gym
from environments.broken_joint import BrokenJointEnv
from environments.rendered_cheetah import CheetahEnv
from environments.stacked import StackedEnv
from models.darc import DARC
from environments.rendered_reacher import RenderedReacher

# source_env = RenderedReacher("reacher.xml")
# target_env = RenderedReacher(
#     "/environments/assets/reacher.xml")
source_env = StackedEnv(CheetahEnv(
    '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah.xml'), 20, 50,
    4, 2)
target_env = StackedEnv(BrokenJointEnv(CheetahEnv(
    '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah2.xml'), [1]), 20, 50,
    4, 2)
# env._max_episode_steps = 3000
state_dim = source_env.observation_space.shape[0]
action_dim = source_env.action_space.shape[0]

embedding_size = 256
policy_config = {
    "input_dim": [state_dim],
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 3, 'stride': 2},
                     {"name": "conv2", "channels": 32, 'kernel_size': 3, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 2},
                     {"name": "linear1", "size": embedding_size},
                     {"name": "linear1", "size": 256},
                     {"name": "linear2", "size": 256},
                     {"name": "split1", "sizes": [action_dim, action_dim]}],
    "hidden_activation": "relu",
    "output_activation": None
}
value_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 256},
                     {"name": "linear2", "size": 256},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": "relu",
    "output_activation": None
}
sa_config = {
    "input_dim": [state_dim + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": None
}
sas_config = {
    "input_dim": [state_dim * 2 + action_dim],
    "architecture": [{"name": "linear1", "size": 64},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": "relu",
    "output_activation": None
}
model = DARC(policy_config, value_config, sa_config, sas_config, source_env, target_env, ent_adj=True,
             n_updates_per_train=2, warmup_games=10)
# model.train(500)
# model.load_model("Ant-v2-DARC-200", "cuda")
model.train(300, deterministic=False)
model.save_model("Ant-v2-DARC2-400")
model.eval(100)
