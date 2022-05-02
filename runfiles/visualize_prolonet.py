import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

def visualize_prolonet(prolonet, raw_indices=False, idx_to_names=None,
                       dont_care_about_weights_smaller_than=0.005,
                       idx_to_actions=None, max_lines=5, node_size=8000, node_color='white', show=True,
                       save_fig_filename=None, save_fig_dimensions=None):

    g = nx.DiGraph()

    node_relabel = {}

    attached_node_indices = set()
    for leaf in prolonet.leaf_init_information:
        for left_parent in leaf[0]:
            attached_node_indices.add(left_parent)
        for right_parent in leaf[1]:
            attached_node_indices.add(right_parent)

    for leaf in attached_node_indices:
        deepest_node = leaf

        weights = prolonet.layers[deepest_node]
        comparator = prolonet.comparators[deepest_node]

        readable_weights = []

        weights_descending = torch.argsort(torch.abs(weights)).detach().numpy()
        for idx in weights_descending[::-1]:
            if abs(float(weights[idx])) > dont_care_about_weights_smaller_than:
                new_str = '({:.2f})'.format(float(weights[idx])) + str(idx_to_names[int(idx)])
                if raw_indices:
                    new_str += '[' + str(int(idx)) + ']'
                readable_weights.append(new_str)

        comparator_string = ''
        num_lines = 0
        for i in range(len(readable_weights)):
            comparator_string += readable_weights[i]
            if num_lines > max_lines:
                comparator_string += '...'
                break
            num_lines += 1
            if i < len(readable_weights) - 1:
                comparator_string += ' +\n'

        comparator_string += '\n > '
        comparator_string += '{:.2f}'.format(float(comparator))

        summary_string = 'Comp ['
        summary_string += str(deepest_node) + ']\n'
        summary_string += comparator_string

        node_relabel[deepest_node] = summary_string

        g.add_node(deepest_node, color=node_color)

    leaf_idx = 0
    actual_leaves = prolonet.action_probs


    for leaf_left, leaf_right, leaf_actions in prolonet.leaf_init_information:
        leaf_actions = actual_leaves[leaf_idx].data.cpu().numpy()

        readable_actions = []
        leaf_actions_descending = np.argsort(np.abs(leaf_actions))
        for i in leaf_actions_descending[::-1]:
            if abs(float(leaf_actions[i])) > dont_care_about_weights_smaller_than:
                new_str = '({:.2f})'.format(leaf_actions[i]) + str(idx_to_actions[i])
                if raw_indices:
                    new_str += '[' + str(int(i)) + ']'
                readable_actions.append(new_str)

        actions_string = 'Action [' + str(leaf_idx) + ']\n'
        num_lines = 0
        for action in readable_actions:
            if num_lines > max_lines:
                actions_string += '...'
                break
            num_lines += 1
            actions_string += str(action) + '\n'

        g.add_node(actions_string, color=node_color)

        for left_parent in leaf_left:
            g.add_edge(left_parent, actions_string, edge_color="green", width=4)

        for right_parent in leaf_right:
            g.add_edge(right_parent, actions_string, edge_color="red", width=2)

        leaf_idx += 1

    g = nx.relabel_nodes(g, node_relabel)

    edge_colors = [g.edges[edge]["edge_color"] for edge in g.edges()]

    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot', args='-Grankdir=LR')
    color_list = [g.nodes[node]["color"] for node in g.nodes()]
    edge_width_list = [g.edges[edge]["width"] for edge in g.edges()]
    nx.draw(g, pos, node_color=color_list, edge_color=edge_colors, width=edge_width_list,
            with_labels=True, node_size=node_size)
    figure = plt.gcf()
    if save_fig_filename != None:
        figure.set_size_inches(save_fig_dimensions)
        plt.savefig(save_fig_filename, dpi=200)
    if show:
        plt.show()

    plt.cla()