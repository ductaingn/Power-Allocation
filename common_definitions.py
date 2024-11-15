"""
Common definitions of variables that can be used across files
"""

from keras.initializers import glorot_normal  # pylint: disable=no-name-in-module

# brain parameters
GAMMA = 0.9  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()

# buffer params
UNBALANCE_P = 0.8  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.3
BATCH_SIZE = 528
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 10000
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4
WARM_UP = 1  # num of warm up epochs

CONFIDENCE = [0.45,0.55]
best_confidence = [0.45,0.55]