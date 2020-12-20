import tensorflow as tf

from environments.broken_joint import BrokenJointEnv
from environments.rendered_cheetah import CheetahEnv
from environments.stacked import StackedEnv
from models.sac import ContSAC

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# target_env = StackedEnv(BrokenJointEnv(CheetahEnv(
#     '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah.xml'), []), 20, 50,
#     4, 2)
target_env = StackedEnv(BrokenJointEnv(CheetahEnv(
    '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah2.xml'), [3]),
    20, 50,
    4, 2)
state_dim = target_env.observation_space.shape
action_dim = target_env.action_space.shape[0]
print(action_dim)

policy_config = {
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "linear2", "size": 512},
                     {"name": "mark_embedding"},
                     {"name": "linear2", "size": 512},
                     {"name": "split1", "sizes": [action_dim, action_dim]}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
value_config = {
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "linear2", "size": 512},
                     {"name": "mark_embedding"},
                     {"name": "linear2", "size": 512},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
model = ContSAC(policy_config, value_config, target_env, ent_adj=True, n_updates_per_train=2, warmup_games=10)
# model.load_model("Embed-Cheetah-SAC-600")
# model.load_embedding("./saved_weights/cheetah-regression-400/embed")
model.train(600, deterministic=False)
# model.save_model("Embed-Cheetah-SAC-600")
# model.eval(400, render=False)
