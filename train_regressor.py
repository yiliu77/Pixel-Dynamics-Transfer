from environments.rendered_cheetah import CheetahEnv
from environments.stacked import StackedEnv
from models.regression import Regression
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

source_env = StackedEnv(CheetahEnv('/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah.xml'), 20, 50, 4, 2)
target_env = StackedEnv(CheetahEnv('/home/yiliu77/Desktop/Python/projects/reinforcement_learning/dirl_darc/environments/assets/cheetah2.xml'), 20, 50, 4, 2)
source_env._max_episode_steps = 9
target_env._max_episode_steps = 9

embed_config = {
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "linear2", "size": 512}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": tf.nn.leaky_relu
}
reg_config = {
    "architecture": [{"name": "linear2", "size": 512},
                     {"name": "linear4", "size": 128},
                     {"name": "linear3", "size": 8}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}
domain_config = {
    "architecture": [{"name": "linear5", "size": 512},
                     {"name": "linear6", "size": 2}],
    "hidden_activation": tf.nn.leaky_relu,
    "output_activation": None
}

model = Regression(embed_config, reg_config, domain_config, source_env, target_env)
# model.train(200)
# model.load_model("regression-400")
model.train(400)
model.save_model("cheetah-regression-400")
model.eval(30)
