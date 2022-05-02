
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

from agents.prolonet_agent import DeepProLoNet
from runfiles.sc_helpers import get_human_readable_mapping, get_human_readable_action_mapping
import visualize_prolonet


if __name__ == '__main__':
    idx_to_name, name_to_idx = get_human_readable_mapping()
    idx_to_action = get_human_readable_action_mapping()
    dim_in = len(idx_to_name)
    dim_out = len(idx_to_action)
    print(dim_in, dim_out)
    bot_name = 'prolo' + '_sc_runner'

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
                                          save_fig_filename='figures/1thprolo_sc_runner_value_network.png',
                                          save_fig_dimensions=(14, 20))

    visualize_prolonet.visualize_prolonet(prolonet=policy_agent.action_network, raw_indices=True,
                                          idx_to_names=idx_to_name,
                                          idx_to_actions=idx_to_action, max_lines=4, node_size=30000,
                                          node_color='#d9d9d9',
                                          show=False,
                                          save_fig_filename='figures/1thprolo_sc_runner_action_network.png',
                                          save_fig_dimensions=(14, 20))
