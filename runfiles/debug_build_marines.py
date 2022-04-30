
import os
import sys
import types

import sc2

sys.path.insert(0, os.path.abspath('../'))

from agents.prolonet_agent import DeepProLoNet
from agents.py_djinn_agent import DJINNAgent
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
from runfiles import build_marines_helpers
import time
import torch.multiprocessing as mp
import argparse
import build_marines_helpers
import numpy as np

FAILED_REWARD = -0.0
STEP_PENALTY = 0.01
SUCCESS_BUILD_REWARD = 0.
SUCCESS_TRAIN_REWARD = 0.
SUCCESS_SCOUT_REWARD = 0.
SUCCESS_ATTACK_REWARD = 0.
SUCCESS_MINING_REWARD = 0.

TYPES = build_marines_helpers.TYPES

class StarmniBot(sc2.BotAI):
    def __init__(self, rl_agent):
        super(StarmniBot, self).__init__()
        self.player_unit_tags = []
        self.agent = rl_agent
        self.corners = None
        self.action_buffer = []
        self.prev_state = None
        # self.last_known_enemy_units = []
        self.itercount = 0
        # self.last_scout_iteration = -100
        # self.scouted_bases = []
        self.last_reward = 0
        # self.num_nexi = 1
        self.mining_reward = SUCCESS_MINING_REWARD
        self.last_sc_action = 43  # Nothing
        self.positions_for_depots = []   # replacing with positions for depots
        self.positions_for_buildings = []
        # self.army_below_half = 0
        # self.last_attack_loop = 0
        self.debug_count = 0

if __name__ == '__main__':
    # TODO: change the agent types to include our own policy network
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='djinn')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-p", "--processes", help="how many processes?", type=int, default=1)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')
    parser.add_argument("-gpu", help="run on GPU?", action='store_true')
    parser.add_argument("-vec", help="Vectorized ProLoNet?", action='store_true')
    parser.add_argument("-adv", help="Adversarial ProLoNet?", action='store_true')
    parser.add_argument("-rand", help="Random ProLoNet?", action='store_true')
    parser.add_argument("-deep", help="Deepening?", action='store_true')
    parser.add_argument("-s", "--sl_init", help="sl to rl for fc net?", action='store_true')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adv  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    NUM_EPS = args.episodes  # num episodes Default 1000
    NUM_PROCS = args.processes  # num concurrent processes Default 1
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    VECTORIZED = args.vec  # Applies for 'prolo' vectorized or no? Default false
    RANDOM = args.rand  # Applies for 'prolo' random init or no? Default false
    DEEPEN = args.deep  # Applies for 'prolo' deepen or no? Default false
    # torch.set_num_threads(NUM_PROCS)
    #dim_in = 14
    dim_in = 30
    dim_out = 10
    bot_name = AGENT_TYPE + 'SC_Macro'+'Medium'
    mp.set_sharing_strategy('file_system')
    if AGENT_TYPE == 'prolo':
        policy_agent = DeepProLoNet(distribution='one_hot',
                                    bot_name=bot_name,
                                    input_dim=dim_in,
                                    output_dim=dim_out,
                                    use_gpu=USE_GPU,
                                    vectorized=VECTORIZED,
                                    randomized=RANDOM,
                                    adversarial=ADVERSARIAL,
                                    deepen=DEEPEN,
                                    deterministic=True)
    elif AGENT_TYPE == 'fc':
        policy_agent = FCNet(input_dim=dim_in,
                             bot_name=bot_name,
                             output_dim=dim_out,
                             sl_init=SL_INIT)
    elif AGENT_TYPE == 'lstm':
        policy_agent = LSTMNet(input_dim=dim_in,
                               bot_name=bot_name,
                               output_dim=dim_out)
    elif AGENT_TYPE == 'djinn':
        policy_agent = DJINNAgent(bot_name=bot_name,
                                  input_dim=dim_in,
                                  output_dim=dim_out)
    else:
        raise Exception('No valid network selected')
    start_time = time.time()

    prev_state = np.zeros((len(TYPES)))
    # prev_state = np.zeros((194))


    prev_state[TYPES.FOOD_CAP.value] = 15

    prev_state[TYPES.SCV.value] = 12


    for i in range(0, 1):

        action = policy_agent.get_action(prev_state)
        print('Action:', action)