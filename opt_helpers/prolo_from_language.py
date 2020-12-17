# Created by Andrew Silva on 7/30/19
import numpy as np
import sys
sys.path.insert(0, '../')
from agents.vectorized_prolonet import ProLoNet
from agents.vectorized_prolonet_helpers import save_prolonet, load_prolonet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from tkinter import *
import copy

global user_command


DOMAIN = 'FireSim'  # options are: 'FireSim' or 'Build_Marines' or 'robot'
FIRE_OPTIONS = [
    'If Fire 1 is to my south ',
    'If Fire 1 is to my north ',
    'If Fire 1 is to my east ',
    'If Fire 1 is to my west ',
    'If I am the closest drone to Fire 1',
    'If Fire 2 is to my south ',
    'If Fire 2 is to my north ',
    'If Fire 2 is to my east ',
    'If Fire 2 is to my west ',
    'If I am the closest drone to Fire 2',
    'Move north',
    'Move south',
    'Move east',
    'Move west'
]

def construct_tree(node_data_in):
    """
    takaes in a data dump from gen_init_params's loop
    :param node_data_in:
    :return:
    """
    weights = []
    comparators = []
    leaves = []
    current_right_path = []
    current_left_path = []
    current_node_index = 0
    # Variables necessary for drawing
    drawing_info = {}
    last_node_leaf = False
    node_x = 0
    node_y = 0
    max_path_len = 1
    for node_index in range(len(node_data_in)):
        data = node_data_in[node_index][0]
        is_leaf = node_data_in[node_index][1]
        command_to_write = node_data_in[node_index][2]
        if len(current_left_path) + len(current_right_path) > max_path_len:
            max_path_len = len(current_left_path)+len(current_right_path)
        if is_leaf:
            leaves.append(
                [current_left_path.copy(), current_right_path.copy(), data])  # paths all set, just throw the leaf on there
            if not current_left_path:  # If we're at the far right end of the tree, we're done.
                drawing_info['lrfinal'] = [command_to_write, node_x, node_y]  # save for drawing
                draw_tree(drawing_info)

                return np.array(weights), np.array(comparators), leaves
            last_node_index = current_node_index - 1
            if last_node_index in current_left_path:  # If from a left split, just back up a step
                current_right_path.append(last_node_index)
                current_left_path.remove(last_node_index)
                new_node_x = node_x + 15 * 2   # * (1 + max_path_len - (len(current_right_path)+len(current_left_path)))
                node_y = node_y
                drawing_info['ll' + str(last_node_index)] = [command_to_write, node_x, node_y, node_x+15, node_y+15, new_node_x, node_y]  # save for drawing
                node_x = new_node_x
            else:  # If from a right split, back alllll the way up to the last left split.
                last_node_index = max(current_left_path)
                prev_path_len = len(current_left_path) + len(current_right_path)
                current_right_path = [i for i in current_right_path if i < last_node_index]
                current_left_path.remove(last_node_index)
                current_right_path.append(last_node_index)
                # TODO: Fix these x / y previous coords...
                new_node_x = drawing_info[last_node_index][1] + (15 * (1 + max_path_len - (len(current_right_path)+len(current_left_path))))
                drawing_node_y = -15*(len(current_left_path)+len(current_right_path)-1) # 15*(len(current_right_path)) - 15*(len(current_left_path)+1)
                drawing_node_x = drawing_info[last_node_index][1]  # 15*len(current_right_path) - 15*(len(current_left_path)+1)
                new_node_y = drawing_info[last_node_index][2] - 15
                drawing_info['lr' + str(last_node_index)] = [command_to_write, node_x, node_y, drawing_node_x, drawing_node_y, new_node_x, new_node_y]  # save for drawing
                node_x = new_node_x
                node_y = new_node_y
            last_node_leaf = True
        else:
            weights.append(data[0])
            comparators.append(data[1])
            current_left_path.append(current_node_index)
            if last_node_leaf:
                new_node_x = node_x - 15
                new_node_y = node_y - 15
            else:
                new_node_x = node_x - 15
                new_node_y = node_y - 15
            drawing_info[current_node_index] = [command_to_write, node_x, node_y, node_x, node_y, new_node_x, new_node_y]
            node_x = new_node_x
            node_y = new_node_y

            current_node_index += 1
            last_node_leaf = False
    draw_tree(drawing_info)
    return False


def gen_init_params():

    data_dump = []
    while True:
        # user_command = get_node_data_from_user()
        gui = make_gui(DOMAIN)

        # start the GUI
        gui.mainloop()
        if user_command == 'Undo':
            data_dump.pop(-1)
        else:
            data, is_leaf = data_from_command(user_command, DOMAIN)
            data_dump.append([data, is_leaf, copy.deepcopy(user_command)])
        finished_outcome = construct_tree(data_dump)
        if finished_outcome:
            return finished_outcome[0], finished_outcome[1], finished_outcome[2]


def make_gui(domain_in):
    if domain_in == 'FireSim':
        gui = Tk()
        button0 = Button(gui, text='Undo', fg='black', bg='white',
                         command=lambda: press(-1, gui, domain_in), height=2, width=30)
        button0.grid(row=1, column=0)

        button1 = Button(gui, text=FIRE_OPTIONS[0], fg='black', bg='white',
                         command=lambda: press(0, gui, domain_in), height=2, width=30)
        button1.grid(row=2, column=0)

        button2 = Button(gui, text=FIRE_OPTIONS[1], fg='black', bg='white',
                         command=lambda: press(1, gui, domain_in), height=2, width=30)
        button2.grid(row=3, column=0)

        button3 = Button(gui, text=FIRE_OPTIONS[2], fg='black', bg='white',
                         command=lambda: press(2, gui, domain_in), height=2, width=30)
        button3.grid(row=4, column=0)

        button4 = Button(gui, text=FIRE_OPTIONS[3], fg='black', bg='white',
                         command=lambda: press(3, gui, domain_in), height=2, width=30)
        button4.grid(row=5, column=0)

        button5 = Button(gui, text=FIRE_OPTIONS[4], fg='black', bg='white',
                         command=lambda: press(4, gui, domain_in), height=2, width=30)
        button5.grid(row=6, column=0)

        button6 = Button(gui, text=FIRE_OPTIONS[5], fg='black', bg='white',
                         command=lambda: press(5, gui, domain_in), height=2, width=30)
        button6.grid(row=7, column=0)

        button7= Button(gui, text=FIRE_OPTIONS[6], fg='black', bg='white',
                         command=lambda: press(6, gui, domain_in), height=2, width=30)
        button7.grid(row=8, column=0)

        button8 = Button(gui, text=FIRE_OPTIONS[7], fg='black', bg='white',
                         command=lambda: press(7, gui, domain_in), height=2, width=30)
        button8.grid(row=9, column=0)

        button9 = Button(gui, text=FIRE_OPTIONS[8], fg='black', bg='white',
                         command=lambda: press(8, gui, domain_in), height=2, width=30)
        button9.grid(row=10, column=0)

        button10 = Button(gui, text=FIRE_OPTIONS[9], fg='black', bg='white',
                         command=lambda: press(9, gui, domain_in), height=2, width=30)
        button10.grid(row=11, column=0)

        button11 = Button(gui, text=FIRE_OPTIONS[10], fg='black', bg='white',
                         command=lambda: press(10, gui, domain_in), height=2, width=30)
        button11.grid(row=1, column=1)

        button12 = Button(gui, text=FIRE_OPTIONS[11], fg='black', bg='white',
                         command=lambda: press(11, gui, domain_in), height=2, width=30)
        button12.grid(row=2, column=1)

        button13 = Button(gui, text=FIRE_OPTIONS[12], fg='black', bg='white',
                         command=lambda: press(12, gui, domain_in), height=2, width=30)
        button13.grid(row=3, column=1)

        button14 = Button(gui, text=FIRE_OPTIONS[13], fg='black', bg='white',
                         command=lambda: press(13, gui, domain_in), height=2, width=30)
        button14.grid(row=4, column=1)
        return gui


def close_event():
    plt.close()  # timer calls this function after 10 seconds and closes the window


def press(int_in, gui_in, domain_in):
    if domain_in == 'FireSim':
        options = FIRE_OPTIONS
    global user_command
    user_command = options[int_in]
    if int_in == -1:
        user_command = 'Undo'
    gui_in.destroy()
    close_event()


def data_from_command(command_in, domain="fire_sim"):
    """
    Interpret some  user command into weights/comparators or a leaf
    :param command_in: text from speech to text?
    :param domain: which domain should I be looking through? fire_sim, sc2, or sawyer
    :return: data (tuple of some sort), is_leaf=True/False
    """
    print(f"Command: {command_in}")
    if domain == 'FireSim':
        weights = np.zeros(6)
        comparator = [5.]
        leaf = np.zeros(5)
        if command_in == FIRE_OPTIONS[0]:  # if fire 1 south
            weights[0] = 1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[1]:  # if fire 1 north
            weights[0] = -1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[2]:  # if fire 1 east
            weights[1] = -1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[3]:  # if fire 1 west
            weights[1] = 1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[4]:  # if closer to fire 1
            weights[4] = 5
            comparator = [1.]
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[5]:  # if fire 2 south
            weights[2] = 1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[6]:  # if fire 2 north
            weights[2] = -1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[7]:  # if fire 2 east
            weights[3] = -1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[8]:  # if fire 2 west
            weights[3] = 1
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[9]:  # if closer to fire 2
            weights[5] = 5
            comparator = [1.]
            return (weights, comparator), False
        elif command_in == FIRE_OPTIONS[10]:  # move north
            leaf[3] = 1
            return leaf, True
        elif command_in == FIRE_OPTIONS[11]:  # move south
            leaf[1] = 1
            return leaf, True
        elif command_in == FIRE_OPTIONS[12]:  # move east
            leaf[2] = 1
            return leaf, True
        elif command_in == FIRE_OPTIONS[13]:  # move west
            leaf[0] = 1
            return leaf, True
    print("Something went wrong...")


def init_actor_and_critic(weights, comparators, leaves, use_gpu=False, alpha=1.0):
    """
    Take in some np arrays of initialization params from gen_init_params
    :param weights:
    :param comparators:
    :param leaves:
    :param use_gpu:
    :param alpha:
    :return:
    """
    dim_in = weights.shape[-1]
    dim_out = leaves[0][-1].shape[-1]
    if len(comparators[0]) == 1:
        vectorized = False
    else:
        vectorized = True
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=weights,
                              comparators=comparators,
                              leaves=leaves,
                              alpha=alpha,
                              is_value=False,
                              use_gpu=use_gpu,
                              vectorized=vectorized)
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=weights,
                             comparators=comparators,
                             leaves=leaves,
                             alpha=alpha,
                             is_value=True,
                             use_gpu=use_gpu,
                             vectorized=vectorized)
    return action_network, value_network


def prolo_from_language_main():
    weights, comparators, leaves = gen_init_params()
    actor, critic = init_actor_and_critic(weights, comparators, leaves, alpha=99999)
    save_prolonet('../study_models/usermadeprolo' + DOMAIN + "_actor_.pth.tar", actor)
    save_prolonet('../study_models/usermadeprolo' + DOMAIN + "_critic_.pth.tar", critic)


def draw_tree(draw_info):
    patches = []

    fig, ax = plt.subplots(figsize=(18, 12))

    timer = fig.canvas.new_timer(interval=10000)  # creating a timer object and setting an interval of 10 sec
    timer.add_callback(close_event)
    lines = []
    text_size = 14
    box_width = 10
    for node in draw_info.values():
        x_coord = node[1]
        y_coord = node[2]
        text = node[0]
        if len(node) > 3:
            l1_x_coord = node[3]
            l1_y_coord = node[4]
            l2_x_coord = node[5]
            l2_y_coord = node[6]
        this_node = mpatches.FancyBboxPatch(xy=[x_coord, y_coord],
                                            width=box_width,
                                            height=1,
                                            boxstyle=mpatches.BoxStyle("Round", pad=4))
        patches.append(this_node)
        # last_node = last_node_coords[node//2]
        line = mlines.Line2D([l1_x_coord+(box_width//2), l2_x_coord+(box_width//2)],
                             [l1_y_coord, l2_y_coord+1])
        lines.append(line)
        plt.text(x_coord+5, y_coord, text, ha="center", family='sans-serif', size=text_size)

    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    for line in lines:
        ax.add_line(line)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.ion()
    # timer.start()
    plt.savefig(f'{DOMAIN}_expert_policy.png')

    plt.show()
    # timer.stop()
    plt.pause(0.01)

prolo_from_language_main()
