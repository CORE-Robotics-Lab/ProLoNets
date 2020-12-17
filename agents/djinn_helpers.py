# Created by Andrew Silva on 10/9/18
import torch
import torch.nn as nn
import numpy as np
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()

DJINN_TREE_DATA = {
    'cart': {
        'feature': np.array([[0], [0], [2], [-2], [-2], [2], [-2], [1], [2], [-2], [-2], [3],
                             [-2], [-2], [2], [1], [3], [-2], [-2], [2], [-2], [-2], [-2]]),
        'children_left': np.array([1, 2, 3, -1, -1, 6, -1, 8, 9, -1, -1, 12, -1, -1, 15, 16, 17, -1, -1, 20, -1, -1, -1]),
        'children_right': np.array([14, 5, 4, -1, -1, 7, -1, 11, 10, -1, -1, 13, -1, -1, 22, 19, 18, -1, -1, 21, -1, -1, -1])

        # 'children_right': np.array(
        #     [1, 2, 3, -1, -1, 6, -1, 8, 9, -1, -1, 12, -1, -1, 15, 16, 17, -1, -1, 20, -1, -1, -1]),
        # 'children_left': np.array(
        #     [14, 5, 4, -1, -1, 7, -1, 11, 10, -1, -1, 13, -1, -1, 22, 19, 18, -1, -1, 21, -1, -1, -1])
    },
    'lunar': {
        'feature': np.array([[1], [3], [5], [6, 7], [-2], [4], [-2], [-2], [-2], [6, 7], [-2], [0], [-2], [0], [-2],
                             [-2], [5], [5], [6, 7], [-2], [-2], [0], [-2], [0], [-2], [-2], [6, 7], [-2], [-2]]),
        'children_left': np.array([1, 2, 3, 4, -1, 6, -1, -1, -1, 10, -1, 12, -1, 14, -1, -1,
                                   17, 18, 19, -1, -1, 22, -1, 24, -1, -1, 27, -1, -1]),
        'children_right': np.array([16, 9, 8, 5, -1, 7, -1, -1, -1, 11, -1, 13, -1, 15, -1,
                                    -1, 26, 21, 20, -1, -1, 23, -1, 25, -1, -1, 28, -1, -1])
        # 'children_right': np.array([1, 2, 3, 4, -1, 6, -1, -1, -1, 10, -1, 12, -1, 14, -1,
        #                             -1, 17, 18, 19, -1, -1, 22, -1, 24, -1, -1, 27, -1, -1]),
        # 'children_left': np.array([16, 9, 8, 5, -1, 7, -1, -1, -1, 11, -1, 13, -1, 15, -1,
        #                            -1, 26, 21, 20, -1, -1, 23, -1, 25, -1, -1, 28, -1, -1])

    },
    'sc_micro': {
        'feature': np.array([[14], [-2], [1], [0], [-2], [0], [-2], [-2], [1], [-2], [0], [-2], [-2]]),
        # 'children_left': np.array([1, -1, 3, 4, -1, 6, -1, -1, 9, -1, 11, -1, -1]),
        # 'children_right': np.array([2, -1, 8, 5, -1, 7, -1, -1, 10, -1, 12, -1, -1])
        'children_right': np.array([1, -1, 3, 4, -1, 6, -1, -1, 9, -1, 11, -1, -1]),
        'children_left': np.array([2, -1, 8, 5, -1, 7, -1, -1, 10, -1, 12, -1, -1])

    },
    'sc_macro': {
        'feature': np.array([[10, 12], [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                        62, 63, 64, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                                        96, 97, 98, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                                        130, 131, 132, 133, 134, 135, 136, 137, 138],
                             [-2], [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 99, 100,
                                    101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                                    117, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
                                    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
                                    170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181],
                             [-2], [-2], [4], [-2], [3, 2], [-2], [9, 157], [-2], [30, 178], [38, 186], [22, 170], [-2],
                             [-2], [38, 186], [-2], [-2], [31, 179], [10, 158], [34, 182], [-2], [-2], [-2], [-2]]),
        'children_left': np.array([1, 2, -1, 4, -1, -1, 7, -1, 9, -1, 11, -1, 13, 14, 15,
                                   -1, -1, 18, -1, -1, 21, 22, 23, -1, -1, -1, -1]),
        'children_right': np.array([6, 3, -1, 5, -1, -1, 8, -1, 10, -1, 12, -1, 20, 17,
                                    16, -1, -1, 19, -1, -1, 26, 25, 24, -1, -1, -1, -1])
    },
    'fire': {
        'feature': np.array([[4], [0], [-2], [0], [-2], [1], [-2], [1], [-2], [-2], [2],
                             [-2], [2], [-2], [3], [-2], [3], [-2], [-2]]),
        # 'children_left': np.array([1, 2, -1, 4, -1, 6, -1, 8, -1, -1, 11, -1, 13, -1, 15, -1, 17, -1, -1]),
        # 'children_right': np.array([10, 3, -1, 5, -1, 7, -1, 9, -1, -1, 12, -1, 14, -1, 16, -1, 18, -1, -1])
        'children_right': np.array([1, 2, -1, 4, -1, 6, -1, 8, -1, -1, 11, -1, 13, -1, 15, -1, 17, -1, -1]),
        'children_left': np.array([10, 3, -1, 5, -1, 7, -1, 9, -1, -1, 12, -1, 14, -1, 16, -1, 18, -1, -1])
    }
}


def save_checkpoint(djinn_model, filename='djinn.pth.tar'):
    """
    Helper function to save checkpoints of PyTorch models
    :param state: Everything to save with the checkpoint (params/weights, optimizer state, epoch, loss function, etc.)
    :param filename: Filename to save the checkpoint under
    :return: None
    """
    if type(djinn_model) is not list:
        djinn_model = [djinn_model]
    master_list = []
    for model in djinn_model:
        state_dict = model.state_dict()
        master_list.append(state_dict)
    torch.save(master_list, filename)


def load_checkpoint(filename='tree0.pth.tar', drop_prob=0.0):
    if not use_gpu:
        model_checkpoint_master = torch.load(filename, map_location='cpu')
    else:
        model_checkpoint_master = torch.load(filename)
    master_model_list = []
    for model_checkpoint in model_checkpoint_master:
        num_layers = len(model_checkpoint.keys())//2
        fake_weights = []
        for i in range(num_layers):
            fake_weights.append(np.random.random_sample([4, 4]))
        new_net = PyDJINN(5, fake_weights, fake_weights)
        for index in range(0, num_layers-1):
            # weight_key = 'weight'+str(index)
            # bias_key = 'bias'+str(index)
            weight_key_value_pair = model_checkpoint.popitem(last=False)
            bias_key_value_pair = model_checkpoint.popitem(last=False)
            new_layer = nn.Linear(4, 4)
            new_layer.weight.data = weight_key_value_pair[1]
            new_layer.bias.data = bias_key_value_pair[1]
            new_seq = nn.Sequential(
                new_layer,
                nn.ReLU(),
                nn.Dropout(drop_prob)
            )
            new_net.layers[index] = new_seq
        new_net.final_layer.weight.data = model_checkpoint['final_layer.weight']
        new_net.final_layer.bias.data = model_checkpoint['final_layer.bias']
        master_model_list.append(new_net)
    return master_model_list

def xavier_init(dim_in, dim_out):
    dist = np.random.normal(0.0, scale=np.sqrt(3.0/(dim_in+dim_out)))
    return dist


def tree_to_nn_weights(dim_in, dim_out, tree_dict):
    """
    :param dim_in: input data (batch first)
    :param dim_out: output data (batch first)
    :param tree_dict: dictionary of features, left children, right children.
    :return:
    """

    tree_to_net = {
        'input_dim': dim_in,
        'output_dim': dim_out,
        'net_shape': {},
        'weights': {},
        'biases': {}
    }

    features = tree_dict['feature']
    children_left = tree_dict['children_left']
    children_right = tree_dict['children_right']
    num_nodes = len(features)

    node_depth = np.zeros(num_nodes, dtype=np.int64)
    is_leaves = np.zeros(num_nodes, dtype=np.int64)
    stack = [(0, -1)]  # node id and parent depth of the root (0th id, no parents means -1...)
    while len(stack) > 0:
        # Recurse through all nodes, adding all left nodes and then right nodes to the stack
        # (we go all the way left before going down any right,then we go right from the bottom-up)
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth+1))
            stack.append((children_right[node_id], parent_depth+1))
        else:
            # If left == right, they're both -1 and it's a leaf
            is_leaves[node_id] = 1

    # This bit appears to just recreate the information above, but more centralized (attach all nodes to their depth
    # and make sure they have children info attached)
    node_dict = {}
    for i in range(len(features)):
        node_dict[i] = {}
        node_dict[i]['depth'] = node_depth[i]
        if np.any([f > 0 for f in features[i]]):
        # if features[i] >= 0:
            node_dict[i]['features'] = features[i]
        else:
            node_dict[i]['features'] = [-2]
        node_dict[i]['child_left'] = features[children_left[i]]
        node_dict[i]['child_right'] = features[children_right[i]]

    num_layers = len(np.unique(node_depth))
    nodes_per_level = np.zeros(num_layers)
    leaves_per_level = np.zeros(num_layers)

    # For each layer, get number of nodes at that layer. If feature is below zero, it's a leaf. Otherwise, it's a node.
    for i in range(num_layers):
        ind = np.where(node_depth == i)[0]
        all_feats = []
        for f in features[ind]:
            all_feats.extend(f)
        all_feats = np.array(all_feats)
        nodes_per_level[i] = len(np.where(all_feats >= 0)[0])
        leaves_per_level[i] = len(np.where(all_feats < 0)[0])

    max_depth_feature = np.zeros(dim_in)
    # Find the deepest feature... for some reason?
    for i in range(len(max_depth_feature)):
        ind = np.where(features == i)[0]
        if len(ind) > 0:
            max_depth_feature[i] = np.max(node_depth[ind])

    djinn_arch = np.zeros(num_layers, dtype=np.int64)

    djinn_arch[0] = dim_in
    for i in range(1, num_layers):
        djinn_arch[i] = djinn_arch[i-1] + nodes_per_level[i]
    djinn_arch[-1] = dim_out

    djinn_weights = {}
    for i in range(num_layers-1):
        djinn_weights[i] = np.zeros((djinn_arch[i+1], djinn_arch[i]))

    new_indices = []
    for i in range(num_layers-1):
        input_dim = djinn_arch[i]
        output_dim = djinn_arch[i+1]
        new_indices.append(np.arange(input_dim, output_dim))
        for f in range(dim_in):
            if i < max_depth_feature[f]-1:
                djinn_weights[i][f, f] = 1.0
        input_index = 0
        output_index = 0
        for index, node in node_dict.items():
            if node['depth'] != i or node['features'][0] < 0:
                continue
            feature = node['features']
            left = node['child_left']
            right = node['child_right']
            if index == 0 and (left[0] < 0 or right[0] < 0):
                for j in range(i, num_layers-2):
                    djinn_weights[j][feature, feature] = 1.0
                djinn_weights[num_layers-2][:, feature] = 1.0
            if left[0] >= 0:
                if i == 0:
                    djinn_weights[i][new_indices[i][input_index],
                                     feature] = xavier_init(input_dim, output_dim)
                else:
                    djinn_weights[i][new_indices[i][input_index],
                                     new_indices[i-1][output_index]] = xavier_init(input_dim, output_dim)

                djinn_weights[i][new_indices[i][input_index], left] = xavier_init(input_dim, output_dim)
                input_index += 1
                # TODO: comment below?
                if output_index >= len(new_indices[i-1]):
                    output_index = 0

            if left[0] < 0 and index != 0:
                leaf_ind = new_indices[i-1][output_index]
                for j in range(i, num_layers-2):
                    djinn_weights[j][leaf_ind, leaf_ind] = 1.0
                djinn_weights[num_layers-2][:, leaf_ind] = 1.0

            if right[0] >= 0:
                if i == 0:
                    djinn_weights[i][new_indices[i][input_index],
                                     feature] = xavier_init(input_dim, output_dim)
                else:
                    djinn_weights[i][new_indices[i][input_index],
                                     new_indices[i-1][output_index]] = xavier_init(input_dim, output_dim)

                djinn_weights[i][new_indices[i][input_index], right] = xavier_init(input_dim, output_dim)
                input_index += 1
                if output_index >= len(new_indices[i-1]):
                    output_index = 0

            if right[0] < 0 and index != 0:
                leaf_ind = new_indices[i-1][output_index]
                for j in range(i, num_layers-2):
                    djinn_weights[j][leaf_ind:leaf_ind] = 1.0
                djinn_weights[num_layers-2][:, leaf_ind] = 1.0
            output_index += 1

    m = len(new_indices[-2])
    ind = np.where(abs(djinn_weights[num_layers-3][:, -m:]) > 0)[0]
    for indices in range(len(djinn_weights[num_layers-2][:, ind])):
        djinn_weights[num_layers-2][indices, ind] = xavier_init(input_dim, output_dim)

    # Convert into a single array because we're not ensembling
    n_hidden = {}
    for i in range(1, len(djinn_arch) - 1):
        n_hidden[i] = djinn_arch[i]

    # existing weights from above, biases could be improved... ignoring for now
    w = []
    b = []
    for i in range(0, len(djinn_arch) - 1):
        w.append(djinn_weights[i].astype(np.float32))
        # b.append(xavier_init(n_hidden[i], n_hidden[i+1]))  # This might be wrong
    tree_to_net['net_shape'] = djinn_arch
    tree_to_net['weights'] = w
    tree_to_net['biases'] = []  # biases?

    return tree_to_net


class PyDJINN(nn.Module):
    def __init__(self, input_dim, weights, biases, drop_prob=0.5, is_value=False):
        super(PyDJINN, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.layers.append(nn.Linear(input_dim, weights[0].shape[0]))
        weight_inits = torch.Tensor(weights[0])
        weight_inits.requires_grad = True
        self.layers[0].weight.data = weight_inits
        # self.layers[0].bias.data.fill_(biases[0])
        last_dim = weights[0].shape[0]
        for index in range(1, len(weights)-1):
            new_linear_layer = nn.Linear(last_dim, weights[index].shape[0])
            weight_inits = torch.Tensor(weights[index])
            weight_inits.requires_grad = True
            new_linear_layer.weight.data = weight_inits
            # new_linear_layer.bias.data.fill_(biases[index])
            new_layer = nn.Sequential(
                new_linear_layer,
                nn.ReLU(),
                nn.Dropout(drop_prob)
            )
            self.layers.append(new_layer)
            last_dim = weights[index].shape[0]
        self.final_layer = nn.Linear(last_dim, weights[-1].shape[0])
        weight_inits = torch.Tensor(weights[-1])
        weight_inits.requires_grad = True
        self.final_layer.weight.data = weight_inits
        # self.final_layer.bias.data.fill_(biases['out'])
        self.softmax = nn.Softmax(dim=1)
        self.is_value = is_value

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
            # print(input_data)
        if self.is_value:
            return self.final_layer(input_data)
        else:
            return self.softmax(self.final_layer(input_data))
