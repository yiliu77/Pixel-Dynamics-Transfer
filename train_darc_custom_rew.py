import tensorflow as tf

from environments.broken_joint import BrokenJointEnv
from environments.rendered_cheetah import CheetahEnv
from environments.stacked import StackedEnv
from models.darc_custom_rew_double_sac import DARCCustomRew2

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

source_env = StackedEnv(BrokenJointEnv(CheetahEnv(
    '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah.xml'), []), 20, 50,
    4, 2)
target_env = StackedEnv(BrokenJointEnv(CheetahEnv(
    '/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah.xml'), [3]), 20, 50,
    4, 2)

state_dim = source_env.observation_space.shape
action_dim = source_env.action_space.shape[0]

policy_config = {
    "architecture": [{"name": "linear2", "size": 128},
                     {"name": "linear2", "size": 64},
                     {"name": "split1", "sizes": [action_dim, action_dim]}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
value_config = {
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "linear2", "size": 32},
                     {"name": "mark_embedding"},
                     {"name": "linear2", "size": 64},
                     {"name": "linear2", "size": 64},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
embedding_config = {
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "linear2", "size": 32}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
s_config = {
    "architecture": [{"name": "linear2", "size": 32},
                     {"name": "linear2", "size": 32},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
sa_config = {
    "architecture": [{"name": "linear2", "size": 32},
                     {"name": "linear2", "size": 32},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
sas_config = {
    "architecture": [{"name": "linear2", "size": 32},
                     {"name": "linear2", "size": 32},
                     {"name": "linear2", "size": 2}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
model = DARCCustomRew2(policy_config, value_config, embedding_config, s_config, sa_config, sas_config, source_env,
                    target_env, log_dir="darc_custom_rew_cheetah", ent_adj=True, n_updates_per_train=2, warmup_games=10)
# model.load_model("Cheetah-DARC-Custom-Rew-Embedding-Space-300")
model.train(300, deterministic=False)
# model.save_model("Cheetah-DARC-Custom-Rew-Embedding-Space-300")
# model.eval(100)
