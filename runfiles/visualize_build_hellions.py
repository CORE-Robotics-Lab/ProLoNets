
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from agents.prolonet_agent import DeepProLoNet
import build_other_units_helpers
import visualize_prolonet


if __name__ == '__main__':
    idx_to_name, name_to_idx = build_other_units_helpers.get_human_readable_mapping()
    idx_to_action = build_other_units_helpers.get_human_readable_action_mapping()
    dim_in = len(idx_to_name)
    dim_out = len(idx_to_action)
    bot_name = 'prolo' + '_hellions'

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

    load_result = policy_agent.load(fn_botname='../../../good_models/server_run/1000thprolo_hellions_gpu')
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

    load_result = policy_agent.load(fn_botname='../../../good_models/server_run/100thprolo_hellions_rand_gpu')
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

    load_result = policy_agent.load(fn_botname='../models/100thprolo_hellions')
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

    load_result = policy_agent.load(fn_botname='../models/1000thprolo_hellions')
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

