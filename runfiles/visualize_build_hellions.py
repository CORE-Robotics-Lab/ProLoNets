
import os
import sys

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
import build_other_units_helpers
import sc_helpers
import numpy as np

import visualize_prolonet

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
    #dim_in = 14
    idx_to_name, name_to_idx = build_other_units_helpers.get_human_readable_mapping()
    print(len(idx_to_name))
    idx_to_action = build_other_units_helpers.get_human_readable_action_mapping()
    print(len(idx_to_action))
    dim_in = len(idx_to_name)
    dim_out = len(idx_to_action)
    bot_name = 'prolo' + '_hellions'
    mp.set_sharing_strategy('file_system')

    policy_agent = DeepProLoNet(distribution='one_hot',
                                bot_name=bot_name,
                                input_dim=dim_in,
                                output_dim=dim_out,
                                use_gpu=False,
                                vectorized=False,
                                randomized=False,
                                adversarial=False,
                                deepen=False,
                                deterministic=True)

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.value_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='../figures/1thprolo_hellions_value_network.png',
                                          save_fig_dimensions=(14, 15))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='../figures/1thprolo_hellions_action_network.png',
                                          save_fig_dimensions=(14, 15))

    load_result = policy_agent.load_filename('../../../good_models/server_run/1000thprolo_hellions_gpu')
    if not load_result:
        print("model file not found")
        quit()

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.value_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=50000,
                                          show=False,
                                          node_color='#d9d9d9',
                                          save_fig_filename='../figures/1000thprolo_hellions_value_network.png',
                                          save_fig_dimensions=(18, 15))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=50000, show=False,
                                          node_color='#d9d9d9',
                                          save_fig_filename='../figures/1000thprolo_hellions_action_network.png',
                                          save_fig_dimensions=(18, 15))

    load_result = policy_agent.load_filename('../../../good_models/server_run/100thprolo_hellions_rand_gpu')
    if not load_result:
        print("model file not found")
        quit()

    width = 20
    height = 30

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.value_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='../figures/100thprolo_hellions_rand_value_network.png',
                                          save_fig_dimensions=(width, height))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='../figures/100thprolo_hellions_rand_action_network.png',
                                          save_fig_dimensions=(width, height))

    load_result = policy_agent.load_filename('../../../good_models/server_run/1000thprolo_hellions_rand_gpu')
    if not load_result:
        print("model file not found")
        quit()

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.value_network, raw_indices=True, idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=50000,
                                          show=False,
                                          node_color='#d9d9d9',
                                          save_fig_filename='../figures/1000thprolo_hellions_rand_value_network.png',
                                          save_fig_dimensions=(width, height))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=50000, show=False,
                                          node_color='#d9d9d9',
                                          save_fig_filename='../figures/1000thprolo_hellions_rand_action_network.png',
                                          save_fig_dimensions=(width, height))

    load_result = policy_agent.load_filename('../models/100thprolo_hellions')
    if not load_result:
        print("model file not found")
        quit()

    width = 18
    height = 15

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.value_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='../figures/100thprolo_hellions_alex_run_value_network.png',
                                          save_fig_dimensions=(width, height))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='../figures/100thprolo_hellions_alex_run_action_network.png',
                                          save_fig_dimensions=(width, height))

    load_result = policy_agent.load_filename('../models/1000thprolo_hellions')
    if not load_result:
        print("model file not found")
        quit()

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.value_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=50000,
                                          show=False,
                                          node_color='#d9d9d9',
                                          save_fig_filename='../figures/1000thprolo_hellions_alex_run_value_network.png',
                                          save_fig_dimensions=(width, height))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=50000, show=False,
                                          node_color='#d9d9d9',
                                          save_fig_filename='../figures/1000thprolo_hellions_alex_run_action_network.png',
                                          save_fig_dimensions=(width, height))

